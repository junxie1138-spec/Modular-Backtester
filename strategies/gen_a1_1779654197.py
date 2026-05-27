from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class ShockwaveParams:
    streak_n: int = 4
    atr_mult: float = 2.0


class GeneratedStrategy(BaseStrategy[ShockwaveParams]):
    strategy_id = "gen_a1_1779654197"

    @classmethod
    def params_type(cls):
        return ShockwaveParams

    def warmup_bars(self, params: ShockwaveParams) -> int:
        return max(int(params.streak_n) + 1, 14) + 1

    def indicators(self, data: pd.DataFrame, params: ShockwaveParams) -> pd.DataFrame:
        high = data["high"]
        low = data["low"]
        close = data["close"]

        rng = high - low

        narrows = (rng < rng.shift(1)).fillna(False).astype(int)
        reset = (narrows == 0).cumsum()
        streak = narrows.groupby(reset).cumsum().astype(float)

        n = int(params.streak_n)
        streak_max_high = high.rolling(n, min_periods=n).max()

        prev_close = close.shift(1)
        tr = pd.concat(
            [
                (high - low),
                (high - prev_close).abs(),
                (low - prev_close).abs(),
            ],
            axis=1,
        ).max(axis=1)
        atr = tr.rolling(14, min_periods=14).mean()

        out = pd.DataFrame(
            {
                "streak": streak,
                "streak_max_high": streak_max_high,
                "atr": atr,
            },
            index=data.index,
        )
        return out

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: ShockwaveParams,
    ) -> SignalFrame:
        close = data["close"].to_numpy(dtype=float)
        high = data["high"].to_numpy(dtype=float)
        streak = indicators["streak"].fillna(0.0).to_numpy(dtype=float)
        streak_max_high = indicators["streak_max_high"].to_numpy(dtype=float)
        atr = indicators["atr"].to_numpy(dtype=float)

        n = int(params.streak_n)
        atr_mult = float(params.atr_mult)
        breakeven_pct = 0.02

        nbars = len(data)
        raw = np.zeros(nbars, dtype=np.int64)

        in_pos = False
        entry_price = 0.0
        stop = 0.0
        breakeven_armed = False

        for i in range(1, nbars):
            if not in_pos:
                prev_streak = streak[i - 1]
                prev_max = streak_max_high[i - 1]
                prev_atr = atr[i - 1]
                if (
                    prev_streak >= n
                    and not np.isnan(prev_max)
                    and not np.isnan(prev_atr)
                    and prev_atr > 0.0
                    and close[i] > prev_max
                ):
                    raw[i] = 1
                    in_pos = True
                    entry_price = close[i]
                    stop = entry_price - atr_mult * prev_atr
                    breakeven_armed = False
            else:
                raw[i] = 1

                if (
                    not breakeven_armed
                    and entry_price > 0.0
                    and high[i] >= entry_price * (1.0 + breakeven_pct)
                ):
                    if entry_price > stop:
                        stop = entry_price
                    breakeven_armed = True

                cur_atr = atr[i]
                if breakeven_armed and not np.isnan(cur_atr):
                    trail = close[i] - atr_mult * cur_atr
                    if trail > stop:
                        stop = trail

                if close[i] <= stop:
                    raw[i] = 0
                    in_pos = False
                    entry_price = 0.0
                    stop = 0.0
                    breakeven_armed = False

        signal = pd.Series(raw, index=data.index, dtype=np.int64)
        signal = signal.shift(1).fillna(0).astype(int)
        size = pd.Series(1.0, index=data.index, dtype=float)
        out = pd.DataFrame({"signal": signal, "size": size}, index=data.index)
        return SignalFrame(data=out, signal_column="signal", size_column="size")
