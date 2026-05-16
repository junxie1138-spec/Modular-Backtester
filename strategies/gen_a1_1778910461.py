from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class GeneratedParams:
    ma_window: int = 200
    mom_window: int = 10
    breakout_window: int = 20
    std_window: int = 20
    atr_window: int = 14
    entry_z: float = 0.75
    rearm_z: float = 0.0
    atr_mult: float = 2.5
    size: float = 1.0


class GeneratedStrategy(BaseStrategy[GeneratedParams]):
    strategy_id = "gen_a1_1778910461"

    @classmethod
    def params_type(cls) -> type[GeneratedParams]:
        return GeneratedParams

    @staticmethod
    def warmup_bars(params: GeneratedParams) -> int:
        return int(
            params.ma_window
            + params.mom_window
            + params.breakout_window
            + 5
        )

    @staticmethod
    def indicators(data: pd.DataFrame, params: GeneratedParams) -> pd.DataFrame:
        close = data["close"]
        high = data["high"]
        low = data["low"]

        ret = close.pct_change()
        mom = ret.rolling(params.mom_window).sum()

        ret_std = ret.rolling(params.std_window).std()
        scale = ret_std * np.sqrt(float(params.mom_window))
        scale = scale.replace(0.0, np.nan)
        mom_z = mom / scale

        # Breakout reference: highest cumulative-return over the prior window.
        mom_max_prev = mom.rolling(params.breakout_window).max().shift(1)

        ma = close.rolling(params.ma_window).mean()

        prev_close = close.shift(1)
        tr = pd.concat(
            [
                (high - low),
                (high - prev_close).abs(),
                (low - prev_close).abs(),
            ],
            axis=1,
        ).max(axis=1)
        atr = tr.rolling(params.atr_window).mean()

        out = pd.DataFrame(index=data.index)
        out["mom"] = mom
        out["mom_z"] = mom_z
        out["mom_max_prev"] = mom_max_prev
        out["ma"] = ma
        out["atr"] = atr
        return out

    @staticmethod
    def generate_signals(
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: GeneratedParams,
    ) -> SignalFrame:
        close = data["close"].to_numpy(dtype=float)
        mom = indicators["mom"].to_numpy(dtype=float)
        mom_z = indicators["mom_z"].to_numpy(dtype=float)
        mom_max_prev = indicators["mom_max_prev"].to_numpy(dtype=float)
        ma = indicators["ma"].to_numpy(dtype=float)
        atr = indicators["atr"].to_numpy(dtype=float)

        n = len(close)
        raw = np.zeros(n, dtype=np.int64)

        in_pos = False
        armed = True  # hysteresis: must cool below rearm_z after a trade
        high_water = np.nan

        for i in range(n):
            c = close[i]
            valid = (
                np.isfinite(ma[i])
                and np.isfinite(atr[i])
                and np.isfinite(mom[i])
                and np.isfinite(mom_z[i])
                and np.isfinite(mom_max_prev[i])
            )
            if not valid:
                raw[i] = 1 if in_pos else 0
                continue

            if in_pos:
                if c > high_water:
                    high_water = c
                stop = high_water - params.atr_mult * atr[i]
                if c < stop:
                    in_pos = False
                    high_water = np.nan
                    raw[i] = 0
                else:
                    raw[i] = 1
            else:
                # Re-arm only once momentum has genuinely cooled.
                if (not armed) and mom_z[i] < params.rearm_z:
                    armed = True

                regime = c > ma[i]
                breakout = mom[i] > mom_max_prev[i]
                strong = mom_z[i] > params.entry_z

                if armed and regime and breakout and strong:
                    in_pos = True
                    armed = False
                    high_water = c
                    raw[i] = 1
                else:
                    raw[i] = 0

        df = pd.DataFrame(index=data.index)
        df["signal"] = raw
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = float(params.size)
        return SignalFrame(data=df, signal_column="signal", size_column="size")
