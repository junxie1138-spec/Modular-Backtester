from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class CorridorParams:
    range_bars: int = 5
    atr_bars: int = 14
    pressure_bars: int = 3
    entry_pressure: float = 0.10
    min_size: float = 0.20
    max_size: float = 0.95


class GeneratedStrategy(BaseStrategy[CorridorParams]):
    strategy_id = "gen_local_1791343331"

    @classmethod
    def params_type(cls) -> type[CorridorParams]:
        return CorridorParams

    def warmup_bars(self, params: CorridorParams) -> int:
        return int(max(params.atr_bars + 1, params.range_bars + params.pressure_bars))

    def indicators(self, data: pd.DataFrame, params: CorridorParams) -> pd.DataFrame:
        if min(params.range_bars, params.atr_bars, params.pressure_bars) < 1:
            raise ValueError("Lookbacks must be positive")
        if not np.isfinite(params.entry_pressure) or params.entry_pressure <= 0:
            raise ValueError("entry_pressure must be positive and finite")
        if not 0 < params.min_size <= params.max_size <= 1:
            raise ValueError("Sizes must satisfy 0 < min_size <= max_size <= 1")
        out = pd.DataFrame(index=data.index)
        out["previous_close"] = data["close"].shift(1)
        out["true_range"] = pd.concat([
            data["high"] - data["low"],
            (data["high"] - out["previous_close"]).abs(),
            (data["low"] - out["previous_close"]).abs(),
        ], axis=1).max(axis=1).where(out["previous_close"].notna())
        out["atr"] = out["true_range"].rolling(params.atr_bars, min_periods=params.atr_bars).mean()
        out["floor"] = data["low"].rolling(params.range_bars, min_periods=params.range_bars).min()
        out["ceiling"] = data["high"].rolling(params.range_bars, min_periods=params.range_bars).max()
        out["width"] = out["ceiling"] - out["floor"]
        out["floor_change"] = out["floor"].diff()
        out["ceiling_change"] = out["ceiling"].diff()
        out["floor_drift"] = out["floor_change"].rolling(params.pressure_bars, min_periods=params.pressure_bars).mean()
        out["ceiling_drift"] = out["ceiling_change"].rolling(params.pressure_bars, min_periods=params.pressure_bars).mean()
        out["pressure"] = (out["floor_drift"] - out["ceiling_drift"]) / out["atr"].where(out["atr"] > 0)
        out["position"] = (data["close"] - out["floor"]) / out["width"].where(out["width"] > 0)
        return out.replace([np.inf, -np.inf], np.nan)

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: CorridorParams) -> SignalFrame:
        pressure = indicators["pressure"]
        position = indicators["position"]
        valid = np.isfinite(pressure) & np.isfinite(position)
        # The position remains long exactly while the entry predicate holds.
        condition = valid & pressure.gt(params.entry_pressure) & position.between(0.35, 0.85)
        strength = (pressure / (3.0 * params.entry_pressure)).clip(0.0, 1.0).where(condition, 0.0).fillna(0.0)
        df = pd.DataFrame(index=data.index)
        df["signal"] = condition.fillna(False).astype(int)
        df["size"] = params.min_size + (params.max_size - params.min_size) * strength
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = df["size"].shift(1).fillna(params.min_size).astype(float)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


if __name__ == "__main__":
    params = CorridorParams()
    index = pd.date_range("2020-01-01", periods=5)
    data = pd.DataFrame(index=index)
    indicators = pd.DataFrame({
        "pressure": [np.nan, 0.20, 0.25, -0.10, 0.20],
        "position": [0.50] * 5,
    }, index=index)
    result = GeneratedStrategy().generate_signals(data, indicators, None, params).data
    assert result["signal"].tolist() == [0, 0, 1, 1, 0]
    assert np.allclose(result["size"], [0.20, 0.20, 0.70, 0.825, 0.20])
