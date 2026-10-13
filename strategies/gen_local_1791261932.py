from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class BoundaryGapParams:
    lookback: int = 12
    gap_min: float = 0.2
    strength_min: float = 0.6
    hold_bars: int = 2


class GeneratedStrategy(BaseStrategy[BoundaryGapParams]):
    strategy_id = "gen_local_1791261932"

    @classmethod
    def params_type(cls) -> type[BoundaryGapParams]:
        return BoundaryGapParams

    @staticmethod
    def warmup_bars(params: BoundaryGapParams) -> int:
        if params.lookback < 2:
            raise ValueError("lookback must be at least 2")
        return int(params.lookback + 2)

    def indicators(self, data: pd.DataFrame, params: BoundaryGapParams) -> pd.DataFrame:
        self.warmup_bars(params)
        out = pd.DataFrame(index=data.index)
        out["prior_close"] = data["close"].shift(1)
        out["range"] = (data["high"] - data["low"]).clip(lower=0)
        out["prior_range_mean"] = out["range"].rolling(params.lookback).mean().shift(1)
        out["gap"] = data["open"] - out["prior_close"]
        out["gap_score"] = out["gap"] / out["prior_range_mean"].replace(0, np.nan)
        out["high_step"] = data["high"].diff()
        out["low_step"] = data["low"].diff()
        out["signed_travel"] = out["high_step"] + out["low_step"]
        out["absolute_travel"] = out["high_step"].abs() + out["low_step"].abs()
        out["prior_signed_travel"] = out["signed_travel"].rolling(params.lookback).sum().shift(1)
        out["prior_absolute_travel"] = out["absolute_travel"].rolling(params.lookback).sum().shift(1)
        out["boundary_strength"] = out["prior_signed_travel"] / out["prior_absolute_travel"].replace(0, np.nan)
        return out.replace([np.inf, -np.inf], np.nan)

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: BoundaryGapParams,
    ) -> SignalFrame:
        if params.hold_bars not in (1, 2):
            raise ValueError("hold_bars must be 1 or 2")
        if not np.isfinite(params.gap_min) or params.gap_min <= 0:
            raise ValueError("gap_min must be finite and positive")
        if not np.isfinite(params.strength_min) or not 0 < params.strength_min <= 1:
            raise ValueError("strength_min must be in (0, 1]")
        gap = indicators["gap_score"]
        strength = indicators["boundary_strength"]
        valid = np.isfinite(gap) & np.isfinite(strength)
        long_entry = valid & (gap >= params.gap_min) & (strength >= params.strength_min)
        short_entry = valid & (gap <= -params.gap_min) & (strength <= -params.strength_min)
        entries = np.select([long_entry, short_entry], [1, -1], default=0)
        positions = np.zeros(len(data), dtype=int)
        direction = 0
        exit_at = -1
        for i, entry in enumerate(entries):
            if direction != 0:
                if i >= exit_at:
                    direction = 0
            elif entry != 0:
                direction = int(entry)
                exit_at = i + params.hold_bars
            positions[i] = direction
        df = pd.DataFrame({"signal": positions, "size": 0.95}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


if __name__ == "__main__":
    index = pd.date_range("2020-01-01", periods=7)
    data = pd.DataFrame({"open": 100.0, "high": 101.0, "low": 99.0, "close": 100.0, "volume": 1000.0}, index=index)
    strategy = GeneratedStrategy()
    for direction in (-1, 1):
        supplied = pd.DataFrame({"gap_score": float(direction), "boundary_strength": float(direction)}, index=index)
        result = strategy.generate_signals(data, supplied, None, BoundaryGapParams()).data
        assert result["signal"].tolist() == [0, direction, direction, 0, direction, direction, 0]
        supplied.iloc[:] = np.nan
        assert strategy.generate_signals(data, supplied, None, BoundaryGapParams()).data["signal"].eq(0).all()
