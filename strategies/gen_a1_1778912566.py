from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class ElasticPlasticParams:
    ma_window: int = 8
    yield_z: float = 1.5
    release_z: float = 0.25
    base_size: float = 1.0


class GeneratedStrategy(BaseStrategy[ElasticPlasticParams]):
    """Trend-strength long-only strategy on the distance-from-MA z-score.

    A stretch above the moving average is treated like material deformation.
    Below the yield point the stretch is 'elastic' (expected to revert) and is
    ignored. When the z-score crosses up through the yield point the stretch is
    declared 'plastic' - a strengthened trend - and a long is opened. The
    plastic state is latched and held until the deformation relaxes: the
    position exits only when the z-score falls back to the release level, i.e.
    when the entry condition that defined the plastic regime has flipped off.
    """

    strategy_id = "gen_a1_1778912566"

    @classmethod
    def params_type(cls):
        return ElasticPlasticParams

    @staticmethod
    def warmup_bars(params: ElasticPlasticParams) -> int:
        # Only a single rolling window of length ma_window is used; no diff or
        # pct_change precedes it, so ma_window bars of warmup is sufficient.
        return max(2, int(params.ma_window))

    @staticmethod
    def indicators(data: pd.DataFrame, params: ElasticPlasticParams) -> pd.DataFrame:
        w = max(2, int(params.ma_window))
        close = data["close"].astype(float)
        ma = close.rolling(w, min_periods=w).mean()
        sd = close.rolling(w, min_periods=w).std(ddof=0)
        # Guard against a flat window producing a zero divisor.
        sd = sd.replace(0.0, np.nan)
        z = (close - ma) / sd
        out = pd.DataFrame(index=data.index)
        out["ma"] = ma
        out["z"] = z
        return out

    @staticmethod
    def generate_signals(
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: ElasticPlasticParams,
    ) -> SignalFrame:
        z = indicators["z"].to_numpy(dtype=float)
        n = len(z)

        yield_z = float(params.yield_z)
        release_z = float(params.release_z)
        base = float(params.base_size)
        if base <= 0.0:
            base = 1.0

        signal = np.zeros(n, dtype=np.int64)
        size = np.full(n, base, dtype=float)

        in_pos = False
        for i in range(n):
            zi = z[i]
            if not np.isfinite(zi):
                # No valid stretch reading (warmup or flat window): stay flat.
                in_pos = False
                signal[i] = 0
                continue

            if not in_pos:
                # Elastic until the z-score crosses up through the yield point.
                if zi >= yield_z:
                    in_pos = True
                    signal[i] = 1
            else:
                # Hold the plastic regime until the deformation relaxes.
                if zi <= release_z:
                    in_pos = False
                    signal[i] = 0
                else:
                    signal[i] = 1

            if signal[i] == 1:
                # Size by plastic strain: how far beyond the yield point the
                # stretch sits, clamped to a sane positive band.
                strain = (zi / yield_z) if yield_z > 0.0 else 1.0
                if not np.isfinite(strain):
                    strain = 1.0
                strain = min(max(strain, 1.0), 2.0)
                size[i] = base * strain

        df = pd.DataFrame(index=data.index)
        df["signal"] = signal
        df["size"] = size
        # Mandatory one-bar shift: decide on bar N close, fill on bar N+1.
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = df["size"].clip(lower=1e-6)
        return SignalFrame(data=df, signal_column="signal", size_column="size")
