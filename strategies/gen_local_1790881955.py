from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class RangeChangeParams:
    recent_bars: int = 4
    baseline_bars: int = 18
    direction_bars: int = 18
    expansion_ratio: float = 1.20
    efficiency_threshold: float = 0.30


class GeneratedStrategy(BaseStrategy[RangeChangeParams]):
    strategy_id = "gen_local_1790881955"

    @classmethod
    def params_type(cls) -> type[RangeChangeParams]:
        return RangeChangeParams

    def warmup_bars(self, params: RangeChangeParams) -> int:
        return int(max(params.recent_bars + params.baseline_bars, params.direction_bars + 1))

    def indicators(self, data: pd.DataFrame, params: RangeChangeParams) -> pd.DataFrame:
        if min(params.recent_bars, params.baseline_bars, params.direction_bars) < 1:
            raise ValueError("Lookbacks must be positive")
        if not np.isfinite(params.expansion_ratio) or params.expansion_ratio <= 1.0:
            raise ValueError("expansion_ratio must exceed one")
        if not np.isfinite(params.efficiency_threshold) or not 0.0 < params.efficiency_threshold <= 1.0:
            raise ValueError("efficiency_threshold must be in (0, 1]")

        out = pd.DataFrame(index=data.index)
        out["normalized_range"] = (data["high"] - data["low"]) / data["close"].abs().replace(0.0, np.nan)
        out["recent_range"] = out["normalized_range"].rolling(params.recent_bars, min_periods=params.recent_bars).mean()
        out["baseline_range"] = out["normalized_range"].shift(params.recent_bars).rolling(params.baseline_bars, min_periods=params.baseline_bars).mean()
        out["range_ratio"] = out["recent_range"] / out["baseline_range"].replace(0.0, np.nan)
        out["midpoint"] = (data["high"] + data["low"]) / 2.0
        out["midpoint_change"] = out["midpoint"].diff()
        out["displacement"] = out["midpoint_change"].rolling(params.direction_bars, min_periods=params.direction_bars).sum()
        out["path_length"] = out["midpoint_change"].abs().rolling(params.direction_bars, min_periods=params.direction_bars).sum()
        out["efficiency"] = out["displacement"] / out["path_length"].replace(0.0, np.nan)
        return out.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: RangeChangeParams) -> SignalFrame:
        expanded = indicators["range_ratio"].ge(params.expansion_ratio)
        bullish = expanded & indicators["efficiency"].ge(params.efficiency_threshold)
        bearish = expanded & indicators["efficiency"].le(-params.efficiency_threshold)
        qualification = pd.Series(np.select([bullish, bearish], [1, -1], default=0), index=data.index, dtype=int)
        confirmed = qualification.ne(0) & qualification.eq(qualification.shift(1))

        positions = np.zeros(len(data), dtype=int)
        position = 0
        for i, direction in enumerate(qualification.to_numpy()):
            if position != 0:
                if direction == -position:
                    position = 0
            elif confirmed.iat[i]:
                position = int(direction)
            positions[i] = position

        df = pd.DataFrame({"signal": positions, "size": 0.95}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


if __name__ == "__main__":
    params = RangeChangeParams()
    strategy = object.__new__(GeneratedStrategy)
    index = pd.date_range("2020-01-01", periods=8)
    data = pd.DataFrame({"open": 100.0, "high": 101.0, "low": 99.0, "close": 100.0, "volume": 1000.0}, index=index)
    derived = pd.DataFrame({"range_ratio": [2.0] * 8, "efficiency": [0.8, 0.8, 0.0, -0.8, -0.8, -0.8, 0.8, 0.8]}, index=index)
    result = strategy.generate_signals(data, derived, None, params).data
    assert result["signal"].tolist() == [0, 0, 1, 1, 0, -1, -1, 0]
    assert result["size"].gt(0.0).all()
    flat = pd.concat([data] * 5, ignore_index=True)
    flat.index = pd.date_range("2020-01-01", periods=len(flat))
    computed = strategy.indicators(flat, params)
    assert np.isfinite(computed.to_numpy()).all()
    assert strategy.generate_signals(flat, computed, None, params).data["signal"].eq(0).all()
    assert strategy.warmup_bars(params) >= max(params.recent_bars + params.baseline_bars, params.direction_bars + 1)
