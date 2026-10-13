from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class ParticipationParams:
    baseline_window: int = 10
    min_volume_ratio: float = 1.10
    max_range_ratio: float = 0.90
    min_efficiency: float = 0.35


class GeneratedStrategy(BaseStrategy[ParticipationParams]):
    strategy_id = "gen_local_1791273343"

    @classmethod
    def params_type(cls) -> type[ParticipationParams]:
        return ParticipationParams

    def warmup_bars(self, params: ParticipationParams) -> int:
        return int(max(200, params.baseline_window + 2))

    def indicators(self, data: pd.DataFrame, params: ParticipationParams) -> pd.DataFrame:
        previous_close = data["close"].shift(1)
        displacement = data["close"] - previous_close
        true_range = pd.concat([
            data["high"] - data["low"],
            (data["high"] - previous_close).abs(),
            (data["low"] - previous_close).abs(),
        ], axis=1).max(axis=1).where(previous_close.notna())
        volume_baseline = data["volume"].shift(1).rolling(
            params.baseline_window, min_periods=params.baseline_window
        ).mean()
        range_baseline = true_range.shift(1).rolling(
            params.baseline_window, min_periods=params.baseline_window
        ).mean()
        result = pd.DataFrame({
            "previous_close": previous_close,
            "displacement": displacement,
            "true_range": true_range,
            "volume_baseline": volume_baseline,
            "range_baseline": range_baseline,
            "volume_ratio": data["volume"] / volume_baseline.where(volume_baseline > 0),
            "range_ratio": true_range / range_baseline.where(range_baseline > 0),
            "efficiency": displacement / true_range.where(true_range > 0),
            "ma200": data["close"].rolling(200, min_periods=200).mean(),
        }, index=data.index)
        return result.replace([np.inf, -np.inf], np.nan)

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: ParticipationParams,
    ) -> SignalFrame:
        qualified = (
            indicators.notna().all(axis=1)
            & (data["close"] > indicators["ma200"])
            & (data["volume"] > 0)
            & (indicators["displacement"] > 0)
            & (indicators["volume_ratio"] >= params.min_volume_ratio)
            & (indicators["range_ratio"] <= params.max_range_ratio)
            & (indicators["efficiency"] >= params.min_efficiency)
        ).fillna(False)
        df = pd.DataFrame(index=data.index)
        # Remain long only while entry qualification holds; its reversal exits.
        df["signal"] = qualified.astype(int)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = 0.95
        return SignalFrame(data=df, signal_column="signal", size_column="size")
