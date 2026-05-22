from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class GapShockwaveParams:
    gap_window: int = 18
    vol_window: int = 60
    entry_threshold: float = 1.0
    size_scale: float = 0.5
    size_cap: float = 1.5


class GeneratedStrategy(BaseStrategy[GapShockwaveParams]):
    """Overnight-gap flux shockwave.

    Each bar's overnight gap is measured as a percentage of the prior close.
    The rolling sum of those gap percentages is the accumulated 'flow'
    (density wave); the rolling std of the gap percentages is the channel
    'capacity'. Their ratio (scaled by sqrt of the flux window) is a
    Mach-number z-score: how supersonic the accumulated overnight flow is
    relative to its own natural dispersion.

    Symmetric stop-and-reverse: go long when the Mach number exceeds
    +threshold, short when it drops below -threshold. Between the two
    thresholds the position is held (deadband). The exit of a long is
    therefore exactly the mirror entry of a short and vice versa - a
    signal-reversal exit driven by a symmetric entry/exit rule.
    """

    strategy_id = "gen_a1_1778888500"

    @classmethod
    def params_type(cls):
        return GapShockwaveParams

    @staticmethod
    def warmup_bars(params: GapShockwaveParams) -> int:
        return int(max(params.gap_window, params.vol_window)) + 2

    @staticmethod
    def indicators(data: pd.DataFrame, params: GapShockwaveParams) -> pd.DataFrame:
        out = pd.DataFrame(index=data.index)

        w = max(int(params.gap_window), 2)
        v = max(int(params.vol_window), 2)

        prev_close = data["close"].shift(1)
        safe_prev = prev_close.where(prev_close > 0.0)
        gap_pct = (data["open"] - prev_close) / safe_prev

        gap_vol = gap_pct.rolling(v, min_periods=v).std()
        flux = gap_pct.rolling(w, min_periods=w).sum()

        denom = gap_vol * float(np.sqrt(float(w)))
        denom = denom.where(denom > 0.0)
        mach = flux / denom

        out["gap_pct"] = gap_pct
        out["gap_vol"] = gap_vol
        out["flux"] = flux
        out["mach"] = mach.replace([np.inf, -np.inf], np.nan)
        return out

    @staticmethod
    def generate_signals(
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: GapShockwaveParams,
    ) -> SignalFrame:
        n = len(data)
        mach = indicators["mach"].to_numpy(dtype=float)
        thr = abs(float(params.entry_threshold))
        if thr <= 0.0:
            thr = 1e-9

        pos = np.zeros(n, dtype=int)
        cur = 0
        for i in range(n):
            m = mach[i]
            if not np.isfinite(m):
                pos[i] = cur
                continue
            if m > thr:
                cur = 1
            elif m < -thr:
                cur = -1
            pos[i] = cur

        df = pd.DataFrame(index=data.index)
        df["signal"] = pos

        abs_mach = np.abs(np.nan_to_num(mach, nan=0.0, posinf=0.0, neginf=0.0))
        strength = float(params.size_scale) * abs_mach / thr
        size = np.clip(0.5 + strength, 0.5, max(float(params.size_cap), 0.5))
        df["size"] = size.astype(float)

        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")
