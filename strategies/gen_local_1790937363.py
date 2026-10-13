from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class CorridorParams:
    lookback: int = 18
    migration_threshold: float = 0.25
    coherence_threshold: float = 0.45
    minimum_size: float = 0.20
    maximum_size: float = 0.95


class GeneratedStrategy(BaseStrategy[CorridorParams]):
    strategy_id = "gen_local_1790937363"

    @classmethod
    def params_type(cls) -> type[CorridorParams]:
        return CorridorParams

    @staticmethod
    def warmup_bars(params: CorridorParams) -> int:
        return int(params.lookback) + 1

    def indicators(self, data: pd.DataFrame, params: CorridorParams) -> pd.DataFrame:
        window = int(params.lookback)
        high_step = data["high"].diff()
        low_step = data["low"].diff()
        translation = (high_step + low_step) / 2.0
        deformation = (high_step - low_step).abs() / 2.0
        valid = high_step.notna() & low_step.notna()
        up = ((high_step > 0) & (low_step > 0)).astype(float).where(valid)
        down = ((high_step < 0) & (low_step < 0)).astype(float).where(valid)
        movement = (translation.abs() + deformation).rolling(window, min_periods=window).sum()
        migration = translation.rolling(window, min_periods=window).sum().div(movement.where(movement > 0))
        result = pd.DataFrame(index=data.index)
        result["high_step"] = high_step
        result["low_step"] = low_step
        result["translation"] = translation
        result["deformation"] = deformation
        result["movement"] = movement
        result["migration"] = migration
        result["up_coherence"] = up.rolling(window, min_periods=window).mean()
        result["down_coherence"] = down.rolling(window, min_periods=window).mean()
        return result.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: CorridorParams) -> SignalFrame:
        migration = indicators["migration"]
        long_entry = (migration > params.migration_threshold) & (indicators["up_coherence"] >= params.coherence_threshold)
        short_entry = (migration < -params.migration_threshold) & (indicators["down_coherence"] >= params.coherence_threshold)
        entries = np.select([long_entry, short_entry], [1, -1], default=0).astype(int)
        signals = np.zeros(len(data), dtype=int)
        position = 0
        for i, entry in enumerate(entries):
            if position == 0:
                position = int(entry)
            elif entry == -position:
                position = 0
            signals[i] = position
        coherence = indicators[["up_coherence", "down_coherence"]].max(axis=1)
        strength = (migration.abs() * coherence).clip(0.0, 1.0)
        size = params.minimum_size + (params.maximum_size - params.minimum_size) * strength
        df = pd.DataFrame({"signal": signals, "size": size.astype(float)}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = df["size"].shift(1).fillna(params.minimum_size).astype(float)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


if __name__ == "__main__":
    strategy = GeneratedStrategy()
    params = CorridorParams()
    index = pd.date_range("2020-01-01", periods=5)
    data = pd.DataFrame(index=index)
    indicators = pd.DataFrame({"migration": [0.8, 0.8, -0.8, -0.8, 0.8], "up_coherence": [1.0, 1.0, 0.0, 0.0, 1.0], "down_coherence": [0.0, 0.0, 1.0, 1.0, 0.0]}, index=index)
    result = strategy.generate_signals(data, indicators, None, params).data
    assert result["signal"].tolist() == [0, 1, 1, 0, -1]
    assert result["size"].between(params.minimum_size, params.maximum_size).all()
    assert result["size"].iloc[1] > result["size"].iloc[0]
