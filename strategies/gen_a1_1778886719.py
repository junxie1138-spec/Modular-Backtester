from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class DualTrendParams:
    window: int = 12
    entry_t: float = 2.0
    exit_t: float = 1.5
    size_div: float = 4.0
    min_size: float = 0.5
    max_size: float = 1.5


def _trend_t(y: np.ndarray, w: int):
    """Rolling OLS of y on a time ramp. Returns (slope, t_stat) arrays
    aligned to y, with NaN over the warmup region."""
    n = y.shape[0]
    nan_full = np.full(n, np.nan)
    if w < 3 or n < w:
        return nan_full.copy(), nan_full.copy()
    x = np.arange(w, dtype=float)
    xc = x - x.mean()
    Sxx = float((xc * xc).sum())
    if Sxx <= 0.0:
        return nan_full.copy(), nan_full.copy()
    # sliding weighted sum: num[j] = sum_i xc[i] * y[j+i]
    num = np.convolve(y, xc[::-1], mode="valid")  # length n - w + 1
    slope = num / Sxx
    csum = np.concatenate(([0.0], np.cumsum(y)))
    csum2 = np.concatenate(([0.0], np.cumsum(y * y)))
    sy = csum[w:] - csum[:-w]
    sy2 = csum2[w:] - csum2[:-w]
    Syy = sy2 - (sy * sy) / w
    sse = Syy - slope * slope * Sxx
    sse = np.clip(sse, 1e-12, None)
    denom = max(w - 2, 1)
    se = np.sqrt(sse / (denom * Sxx))
    se = np.where(se <= 0.0, np.nan, se)
    t = slope / se
    pad = np.full(w - 1, np.nan)
    return (np.concatenate((pad, slope)), np.concatenate((pad, t)))


class GeneratedStrategy(BaseStrategy[DualTrendParams]):
    strategy_id = "gen_a1_1778886719"

    @classmethod
    def params_type(cls):
        return DualTrendParams

    def warmup_bars(self, params: DualTrendParams) -> int:
        # OBV consumes one diff, then a window-length regression.
        return int(params.window) + 1

    def indicators(self, data: pd.DataFrame, params: DualTrendParams) -> pd.DataFrame:
        close = data["close"].to_numpy(dtype=float)
        vol = data["volume"].to_numpy(dtype=float)
        n = close.shape[0]
        w = int(params.window)

        # On-balance-volume: cumulative sign(price change) * volume.
        dirn = np.zeros(n, dtype=float)
        if n > 1:
            dirn[1:] = np.sign(np.diff(close))
        obv = np.cumsum(dirn * vol)

        slope_p, t_p = _trend_t(close, w)
        slope_o, t_o = _trend_t(obv, w)

        return pd.DataFrame(
            {
                "t_price": t_p,
                "t_obv": t_o,
                "slope_price": slope_p,
                "slope_obv": slope_o,
            },
            index=data.index,
        )

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: DualTrendParams,
    ) -> SignalFrame:
        idx = data.index
        n = len(idx)

        tp = indicators["t_price"].to_numpy(dtype=float)
        tv = indicators["t_obv"].to_numpy(dtype=float)

        entry_t = float(params.entry_t)
        exit_t = float(params.exit_t)

        # Two-primitive AND: both the price-trend and the volume-flow-trend
        # regressions must be significant. NaN comparisons evaluate False.
        entry_long = (tp > entry_t) & (tv > entry_t)
        # Mirror bearish flip drives the signal-reversal exit.
        exit_flip = (tp < -exit_t) & (tv < -exit_t)

        raw = np.zeros(n, dtype=int)
        pos = 0
        for i in range(n):
            if pos == 0:
                if entry_long[i]:
                    pos = 1
            else:
                if exit_flip[i]:
                    pos = 0
            raw[i] = pos

        signal = (
            pd.Series(raw, index=idx).shift(1).fillna(0).astype(int)
        )

        # Size scaled by the weaker of the two trend t-statistics.
        strength = np.minimum(tp, tv)
        size_div = float(params.size_div) if params.size_div != 0.0 else 1.0
        size_vals = np.clip(
            strength / size_div,
            float(params.min_size),
            float(params.max_size),
        )
        size = (
            pd.Series(size_vals, index=idx)
            .fillna(float(params.min_size))
        )
        size = size.where(size > 0.0, float(params.min_size))

        out = pd.DataFrame({"signal": signal, "size": size}, index=idx)
        return SignalFrame(data=out, signal_column="signal", size_column="size")
