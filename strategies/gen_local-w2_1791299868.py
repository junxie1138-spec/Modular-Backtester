from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class PlateauParams:
    peak_window: int = 63
    plateau_window: int = 5
    history_window: int = 126
    depth_percentile: float = 0.8
    compression_percentile: float = 0.2
    hold_bars: int = 4


class GeneratedStrategy(BaseStrategy[PlateauParams]):
    strategy_id = "gen_local-w2_1791299868"

    @classmethod
    def params_type(cls):
        return PlateauParams

    def warmup_bars(self, params: PlateauParams) -> int:
        return int(params.peak_window + params.plateau_window + params.history_window)

    def indicators(self, data: pd.DataFrame, params: PlateauParams) -> pd.DataFrame:
        out = pd.DataFrame(index=data.index)
        close = data["close"].where(data["close"] > 0)
        peak = close.rolling(params.peak_window, min_periods=params.peak_window).max()
        depth = (1.0 - close / peak).clip(lower=0).replace([np.inf, -np.inf], np.nan)
        band = depth.rolling(params.plateau_window, min_periods=params.plateau_window).max() - depth.rolling(params.plateau_window, min_periods=params.plateau_window).min()
        depth_cutoff = depth.shift(1).rolling(params.history_window, min_periods=params.history_window).quantile(params.depth_percentile)
        band_cutoff = band.shift(1).rolling(params.history_window, min_periods=params.history_window).quantile(params.compression_percentile)
        ready = depth.notna() & band.notna() & depth_cutoff.notna() & band_cutoff.notna()
        out["depth"] = depth.fillna(0.0)
        out["band"] = band.fillna(0.0)
        out["depth_cutoff"] = depth_cutoff.fillna(0.0)
        out["band_cutoff"] = band_cutoff.fillna(0.0)
        compressed = ready & (out["band"] <= out["band_cutoff"])
        out["entry"] = compressed & ~compressed.shift(1).fillna(False) & (out["depth"] > 0) & (out["depth"] >= out["depth_cutoff"])
        return out

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: PlateauParams) -> SignalFrame:
        df = pd.DataFrame(index=data.index)
        signal = np.zeros(len(data), dtype=int)
        entries = indicators["entry"].to_numpy(dtype=bool)
        remaining = 0
        for i, enter in enumerate(entries):
            if remaining:
                remaining -= 1
                signal[i] = int(remaining > 0)
            elif enter:
                signal[i] = 1
                remaining = params.hold_bars
        df["signal"] = signal
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = 1.0
        return SignalFrame(data=df, signal_column="signal", size_column="size")
