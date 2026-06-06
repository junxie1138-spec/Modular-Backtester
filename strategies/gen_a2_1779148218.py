from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class MirrorBandParams:
    window: int = 20
    band: float = 0.05
    max_band: float = 0.20
    size_scale: float = 0.5
    use_capacity: bool = True


class GeneratedStrategy(BaseStrategy[MirrorBandParams]):
    """Symmetric drawdown-recovery mirror band.

    Entry condition E: close sits at least `band` below its rolling peak
    (deep enough drawdown) and no deeper than `max_band` (capacity guard
    that rejects falling-knife overflow).

    Exit condition X: close sits at least `band` above its rolling trough -
    the exact geometric mirror of E through the rolling extremes. The exit
    fires only when this entry condition flips to its reflection, i.e. a
    pure signal-reversal exit. The shared `band` threshold makes the
    entry/exit rule symmetric by construction.
    """

    strategy_id = "gen_a2_1779148218"

    @classmethod
    def params_type(cls) -> type[MirrorBandParams]:
        return MirrorBandParams

    @staticmethod
    def warmup_bars(params: MirrorBandParams) -> int:
        return int(max(2, int(params.window))) + 1

    @staticmethod
    def indicators(data: pd.DataFrame, params: MirrorBandParams) -> pd.DataFrame:
        close = data["close"].astype(float)
        w = int(max(2, int(params.window)))

        peak = close.rolling(w, min_periods=w).max()
        trough = close.rolling(w, min_periods=w).min()

        # drawdown depth below the rolling peak (<= 0)
        dd = close / peak.replace(0.0, np.nan) - 1.0
        # run-up depth above the rolling trough (>= 0) - the mirror quantity
        ru = close / trough.replace(0.0, np.nan) - 1.0

        out = pd.DataFrame(index=data.index)
        out["dd"] = dd
        out["ru"] = ru
        return out

    @staticmethod
    def generate_signals(
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: MirrorBandParams,
    ) -> SignalFrame:
        n = len(data)
        dd = indicators["dd"].to_numpy(dtype=float)
        ru = indicators["ru"].to_numpy(dtype=float)

        band = float(params.band)
        max_band = float(params.max_band)
        if max_band < band:
            max_band = band
        use_cap = bool(params.use_capacity)

        raw = np.zeros(n, dtype=np.int64)
        position = 0
        for i in range(n):
            d = dd[i]
            r = ru[i]
            if not np.isfinite(d) or not np.isfinite(r):
                raw[i] = position
                continue
            if position == 0:
                # entry condition E: deep-enough but not overflowed drawdown
                entry = d <= -band
                if use_cap:
                    entry = entry and (d >= -max_band)
                if entry:
                    position = 1
            else:
                # exit condition X: the symmetric mirror of E has flipped on
                if r >= band:
                    position = 0
            raw[i] = position

        sig = pd.Series(raw, index=data.index)

        # conviction-scaled size: deeper drawdowns get larger size
        depth = np.abs(dd)
        scaled = (depth - band) / band if band > 0.0 else np.zeros(n)
        scaled = np.clip(scaled, 0.0, 3.0)
        size = 1.0 + float(params.size_scale) * scaled
        size = np.where(np.isfinite(size), size, 1.0)
        size = np.clip(size, 0.1, 5.0)

        df = pd.DataFrame(index=data.index)
        df["signal"] = sig.shift(1).fillna(0).astype(int)
        df["size"] = size.astype(float)
        return SignalFrame(data=df, signal_column="signal", size_column="size")
