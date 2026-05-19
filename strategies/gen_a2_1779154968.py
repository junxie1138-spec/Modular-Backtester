from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class Params:
    med_len: int = 50
    comp_frac: float = 0.80
    comp_streak: int = 5
    hl_streak: int = 3
    atr_len: int = 14
    k_init: float = 2.0
    be_trigger: float = 0.03
    k_trail: float = 3.0
    max_hold: int = 40


def _consec(cond: pd.Series) -> pd.Series:
    """Running count of consecutive True values; NaN-safe, never returns NaN."""
    c = cond.fillna(False).astype(bool)
    grp = (~c).cumsum()
    return c.groupby(grp).cumsum().astype(float)


class GeneratedStrategy(BaseStrategy[Params]):
    strategy_id = "gen_a2_1779154968"

    @classmethod
    def params_type(cls):
        return Params

    @staticmethod
    def warmup_bars(params: Params) -> int:
        return int(max(params.med_len, params.atr_len)) + 2

    def indicators(self, data: pd.DataFrame, params: Params) -> pd.DataFrame:
        high = data["high"]
        low = data["low"]
        close = data["close"]
        prev_close = close.shift(1)

        rng = high - low
        rng_med = rng.rolling(params.med_len, min_periods=params.med_len).median()
        compressed = rng < (rng_med * params.comp_frac)
        comp_streak = _consec(compressed)

        higher_low = low > low.shift(1)
        hl_streak = _consec(higher_low)

        tr = pd.concat(
            [
                high - low,
                (high - prev_close).abs(),
                (low - prev_close).abs(),
            ],
            axis=1,
        ).max(axis=1)
        atr = tr.rolling(params.atr_len, min_periods=params.atr_len).mean()

        out = pd.DataFrame(index=data.index)
        out["comp_streak"] = comp_streak
        out["hl_streak"] = hl_streak
        out["atr"] = atr
        return out

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: Params,
    ) -> SignalFrame:
        close = data["close"].to_numpy(dtype=float)
        comp = indicators["comp_streak"].to_numpy(dtype=float)
        hl = indicators["hl_streak"].to_numpy(dtype=float)
        atr = indicators["atr"].to_numpy(dtype=float)
        n = len(close)

        raw = np.zeros(n, dtype=np.int64)
        position = 0
        entry_price = 0.0
        stop = 0.0
        armed = False
        bars_held = 0

        for i in range(n):
            a = atr[i]
            if position == 0:
                enter = (
                    np.isfinite(a)
                    and a > 0.0
                    and comp[i] >= params.comp_streak
                    and hl[i] >= params.hl_streak
                )
                if enter:
                    position = 1
                    entry_price = close[i]
                    stop = close[i] - params.k_init * a
                    armed = False
                    bars_held = 0
                    raw[i] = 1
                else:
                    raw[i] = 0
            else:
                bars_held += 1
                exit_now = close[i] <= stop or bars_held >= params.max_hold
                if exit_now:
                    position = 0
                    armed = False
                    raw[i] = 0
                else:
                    if (not armed) and close[i] >= entry_price * (1.0 + params.be_trigger):
                        armed = True
                        if entry_price > stop:
                            stop = entry_price
                    if armed and np.isfinite(a):
                        trail = close[i] - params.k_trail * a
                        if trail > stop:
                            stop = trail
                    raw[i] = 1

        df = pd.DataFrame(index=data.index)
        df["signal"] = pd.Series(raw, index=data.index).shift(1).fillna(0).astype(int)
        df["size"] = 1.0
        return SignalFrame(data=df, signal_column="signal", size_column="size")
