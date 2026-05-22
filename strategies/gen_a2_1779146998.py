from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class Params:
    ac_window: int = 20
    clv_window: int = 5
    ac_thresh: float = 0.15
    clv_thresh: float = 0.55
    atr_window: int = 14
    k_atr: float = 2.5
    max_hold: int = 10


class GeneratedStrategy(BaseStrategy[Params]):
    strategy_id = "gen_a2_1779146998"

    @classmethod
    def params_type(cls):
        return Params

    @staticmethod
    def warmup_bars(params: Params) -> int:
        return int(max(params.ac_window, params.atr_window, params.clv_window) + 2)

    @staticmethod
    def indicators(data: pd.DataFrame, params: Params) -> pd.DataFrame:
        high = data["high"]
        low = data["low"]
        close = data["close"]

        rng = high - low
        # Close-location value: where the close sits inside the bar's range.
        # Degenerate zero-range bars are treated as neutral (mid-range).
        clv = (close - low) / rng.where(rng > 0.0, np.nan)
        clv = clv.fillna(0.5).clip(0.0, 1.0)

        # Lag-1 rolling autocorrelation of the close-location value.
        clv_lag = clv.shift(1)
        clv_ac = clv.rolling(params.ac_window).corr(clv_lag)
        clv_mean = clv.rolling(params.clv_window).mean()

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
        out["clv"] = clv
        out["clv_ac"] = clv_ac
        out["clv_mean"] = clv_mean
        out["atr"] = atr
        return out

    @staticmethod
    def generate_signals(
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: Params,
    ) -> SignalFrame:
        close = data["close"].to_numpy(dtype=float)
        atr = indicators["atr"].to_numpy(dtype=float)
        clv_ac = indicators["clv_ac"].to_numpy(dtype=float)
        clv_mean = indicators["clv_mean"].to_numpy(dtype=float)

        n = len(close)

        # Raw 'infected' condition: positive close-location autocorrelation
        # while closes sit consistently in the upper half of their range.
        # NaN comparisons evaluate False, so warmup bars are safely excluded.
        raw = (clv_ac > params.ac_thresh) & (clv_mean > params.clv_thresh)

        # Two-bar confirmation: the condition must hold on this bar and the
        # bar before it.
        confirmed = np.zeros(n, dtype=bool)
        if n > 1:
            confirmed[1:] = raw[1:] & raw[:-1]

        state = np.zeros(n, dtype=int)
        in_pos = False
        stop = 0.0
        bars_held = 0

        for i in range(n):
            if in_pos:
                bars_held += 1
                # Fixed volatility stop (set once at entry) plus a time cap
                # that keeps the holding horizon in the 1-2 week band.
                if close[i] <= stop or bars_held >= params.max_hold:
                    in_pos = False
                    state[i] = 0
                else:
                    state[i] = 1
            else:
                a = atr[i]
                if confirmed[i] and np.isfinite(a) and a > 0.0:
                    in_pos = True
                    stop = close[i] - params.k_atr * a
                    bars_held = 0
                    state[i] = 1
                else:
                    state[i] = 0

        df = pd.DataFrame(index=data.index)
        df["signal"] = state
        df["size"] = 1.0
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")
