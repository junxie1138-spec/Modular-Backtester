from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy

_ATR_PERIOD = 14
_ENTER_THRESHOLD = 0.62


@dataclass(slots=True)
class TrendRankParams:
    rank_window: int = 18
    atr_mult: float = 1.8


def _spearman_vs_time(arr: np.ndarray) -> float:
    """Spearman rank-correlation of the window's values against a 1..n time index.

    Both rank vectors are permutations of 1..n, so each has the same fixed
    variance n(n^2-1)/12 and the correlation reduces to the scaled covariance.
    """
    n = arr.shape[0]
    if n < 3 or np.isnan(arr).any():
        return np.nan
    order = np.argsort(arr, kind="quicksort")
    ranks = np.empty(n, dtype=np.float64)
    ranks[order] = np.arange(1, n + 1, dtype=np.float64)
    time_ranks = np.arange(1, n + 1, dtype=np.float64)
    mean = (n + 1) / 2.0
    cov = np.sum((ranks - mean) * (time_ranks - mean))
    denom = n * (n * n - 1) / 12.0
    if denom <= 0.0:
        return np.nan
    return float(cov / denom)


class GeneratedStrategy(BaseStrategy[TrendRankParams]):
    strategy_id = "gen_a1_1778914400"

    @classmethod
    def params_type(cls):
        return TrendRankParams

    def warmup_bars(self, params: TrendRankParams) -> int:
        window = max(int(params.rank_window), 5)
        return int(max(window, _ATR_PERIOD) + 2)

    def indicators(self, data: pd.DataFrame, params: TrendRankParams) -> pd.DataFrame:
        close = data["close"]
        high = data["high"]
        low = data["low"]

        window = max(int(params.rank_window), 5)

        trend = close.rolling(window).apply(_spearman_vs_time, raw=True)

        prev_close = close.shift(1)
        tr = pd.concat(
            [
                high - low,
                (high - prev_close).abs(),
                (low - prev_close).abs(),
            ],
            axis=1,
        ).max(axis=1)
        atr = tr.rolling(_ATR_PERIOD).mean()

        out = pd.DataFrame(index=data.index)
        out["trend_strength"] = trend
        out["atr"] = atr
        return out

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: TrendRankParams,
    ) -> SignalFrame:
        close = data["close"].to_numpy(dtype=np.float64)
        trend = indicators["trend_strength"].to_numpy(dtype=np.float64)
        atr = indicators["atr"].to_numpy(dtype=np.float64)
        n = close.shape[0]

        k = float(params.atr_mult)
        signal = np.zeros(n, dtype=np.int64)

        in_pos = False
        hwm = 0.0
        for i in range(n):
            t = trend[i]
            a = atr[i]
            if not in_pos:
                # Hysteresis: only a strongly monotonic climb opens a position;
                # the exit is the ratcheting trail, not a trend-strength drop,
                # so the signal does not oscillate on small trend wobbles.
                if (not np.isnan(t)) and (not np.isnan(a)) and t >= _ENTER_THRESHOLD:
                    in_pos = True
                    hwm = close[i]
                    signal[i] = 1
            else:
                if close[i] > hwm:
                    hwm = close[i]
                if np.isnan(a):
                    in_pos = False
                    signal[i] = 0
                    continue
                stop = hwm - k * a
                if close[i] <= stop:
                    in_pos = False
                    signal[i] = 0
                else:
                    signal[i] = 1

        df = pd.DataFrame(index=data.index)
        df["signal"] = signal
        df["size"] = 1.0
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")
