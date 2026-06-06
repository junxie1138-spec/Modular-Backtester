from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class StreakParams:
    streak_k: int = 3
    exit_streak: int = 1
    profit_target: float = 0.03
    time_stop: int = 5
    vol_window: int = 7
    target_vol: float = 0.012
    size_base: float = 1.0
    enable_short: bool = True


class GeneratedStrategy(BaseStrategy[StreakParams]):
    strategy_id = "gen_a2_1779145515"

    @classmethod
    def params_type(cls) -> type[StreakParams]:
        return StreakParams

    def warmup_bars(self, params: StreakParams) -> int:
        # pct_change (1) feeds the streak; rolling std of returns needs vol_window+1
        return int(params.vol_window) + 1

    def indicators(self, data: pd.DataFrame, params: StreakParams) -> pd.DataFrame:
        close = data["close"]
        ret = close.pct_change()

        # Signed consecutive run-length of close-to-close return signs.
        sign = np.sign(ret).fillna(0.0)
        change = sign.ne(sign.shift())
        grp = change.cumsum()
        run_len = sign.groupby(grp).cumcount() + 1
        signed_streak = (run_len * sign).astype(float)

        vol = ret.rolling(int(params.vol_window)).std()

        out = pd.DataFrame(index=data.index)
        out["ret"] = ret.fillna(0.0)
        out["streak"] = signed_streak.fillna(0.0)
        out["vol"] = vol
        return out

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: StreakParams,
    ) -> SignalFrame:
        close = data["close"].to_numpy(dtype=float)
        streak = indicators["streak"].to_numpy(dtype=float)
        vol = indicators["vol"].to_numpy(dtype=float)
        n = len(close)

        raw = np.zeros(n, dtype=int)

        k = int(params.streak_k)
        exit_k = int(params.exit_streak)
        pt = float(params.profit_target)
        tstop = int(params.time_stop)

        position = 0
        entry_price = 0.0
        bars_held = 0
        block_long = False
        block_short = False

        for i in range(n):
            st = streak[i]
            if np.isnan(st):
                continue

            # Hysteresis re-arm: clear the entry block only once the streak has
            # decayed back through the inner threshold.
            if block_long and st >= -exit_k:
                block_long = False
            if block_short and st <= exit_k:
                block_short = False

            if position == 0:
                if st <= -k and not block_long:
                    position = 1
                    entry_price = close[i]
                    bars_held = 0
                    raw[i] = 1
                elif params.enable_short and st >= k and not block_short:
                    position = -1
                    entry_price = close[i]
                    bars_held = 0
                    raw[i] = -1
            else:
                bars_held += 1
                if entry_price > 0.0:
                    pnl = (close[i] - entry_price) / entry_price * position
                else:
                    pnl = 0.0
                hit_target = pnl >= pt
                hit_time = bars_held >= tstop
                if hit_target or hit_time:
                    if position == 1:
                        block_long = True
                    else:
                        block_short = True
                    position = 0
                    entry_price = 0.0
                    bars_held = 0
                    raw[i] = 0
                else:
                    raw[i] = position

        # Inverse-volatility conviction sizing, NaN-safe.
        vol_safe = np.where(
            np.isnan(vol) | (vol <= 0.0), float(params.target_vol), vol
        )
        size = np.clip(
            float(params.size_base) * float(params.target_vol) / vol_safe,
            0.25,
            3.0,
        )

        df = pd.DataFrame(index=data.index)
        df["signal"] = raw
        df["size"] = size.astype(float)

        # Mandatory one-bar shift: decide on bar N close, fill on N+1.
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)

        return SignalFrame(data=df, signal_column="signal", size_column="size")
