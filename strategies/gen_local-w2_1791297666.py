from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class AbsorptionParams:
    depth_lookback: int = 12
    regime_lookback: int = 126
    volume_lookback: int = 20
    normal_depth: float = 0.025
    stress_depth: float = 0.05
    stress_regime: float = 0.10
    normal_pressure: float = 0.7
    stress_pressure: float = 1.1
    hold_bars: int = 2


class GeneratedStrategy(BaseStrategy[AbsorptionParams]):
    strategy_id = "gen_local-w2_1791297666"

    @classmethod
    def params_type(cls):
        return AbsorptionParams

    def warmup_bars(self, params: AbsorptionParams) -> int:
        return max(params.depth_lookback, params.regime_lookback, params.volume_lookback)

    def indicators(self, data: pd.DataFrame, params: AbsorptionParams) -> pd.DataFrame:
        out = pd.DataFrame(index=data.index)
        short_peak = data["close"].rolling(params.depth_lookback, min_periods=params.depth_lookback).max()
        long_peak = data["close"].rolling(params.regime_lookback, min_periods=params.regime_lookback).max()
        mean_volume = data["volume"].rolling(params.volume_lookback, min_periods=params.volume_lookback).mean()
        bar_range = (data["high"] - data["low"]).replace(0, np.nan)
        close_location = (2 * data["close"] - data["high"] - data["low"]) / bar_range
        out["depth"] = (1 - data["close"] / short_peak).replace([np.inf, -np.inf], np.nan).fillna(0).clip(lower=0)
        out["regime_depth"] = (1 - data["close"] / long_peak).replace([np.inf, -np.inf], np.nan).fillna(0).clip(lower=0)
        out["pressure"] = (close_location * data["volume"] / mean_volume.replace(0, np.nan)).replace([np.inf, -np.inf], np.nan).fillna(0)
        out["ready"] = (short_peak.notna() & long_peak.notna() & mean_volume.notna()).astype(bool)
        return out

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: AbsorptionParams,
    ) -> SignalFrame:
        stressed = indicators["regime_depth"] >= params.stress_regime
        required_depth = np.where(stressed, params.stress_depth, params.normal_depth)
        required_pressure = np.where(stressed, params.stress_pressure, params.normal_pressure)
        entries = (indicators["ready"] & (indicators["depth"] >= required_depth) & (indicators["pressure"] >= required_pressure)).to_numpy()
        signal = np.zeros(len(data), dtype=int)
        exit_at = -1
        for i, enter in enumerate(entries):
            if i == exit_at:
                exit_at = -1
            elif exit_at > i:
                signal[i] = 1
            elif enter:
                signal[i] = 1
                exit_at = i + params.hold_bars
        df = pd.DataFrame(index=data.index)
        df["signal"] = signal
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = 1.0
        return SignalFrame(data=df, signal_column="signal", size_column="size")
