from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class PositionStreakParams:
    range_bars: int = 8
    min_streak: int = 2
    upper_fraction: float = 0.60


class GeneratedStrategy(BaseStrategy[PositionStreakParams]):
    strategy_id = "gen_local_1790930369"

    @classmethod
    def params_type(cls) -> type[PositionStreakParams]:
        return PositionStreakParams

    def warmup_bars(self, params: PositionStreakParams) -> int:
        return int(params.range_bars + params.min_streak + 1)

    def indicators(self, data: pd.DataFrame, params: PositionStreakParams) -> pd.DataFrame:
        if params.range_bars < 2 or params.min_streak < 1:
            raise ValueError("range_bars must be >= 2 and min_streak >= 1")
        if not 0.0 < params.upper_fraction < 1.0:
            raise ValueError("upper_fraction must be between zero and one")
        out = pd.DataFrame(index=data.index)
        out["channel_high"] = data["high"].rolling(params.range_bars, min_periods=params.range_bars).max().shift(1)
        out["channel_low"] = data["low"].rolling(params.range_bars, min_periods=params.range_bars).min().shift(1)
        out["channel_width"] = out["channel_high"] - out["channel_low"]
        out["relative_position"] = ((data["close"] - out["channel_low"]) / out["channel_width"].where(out["channel_width"] > 0.0)).replace([np.inf, -np.inf], np.nan)
        out["upper_zone"] = out["relative_position"].notna() & out["relative_position"].ge(params.upper_fraction)
        out["occupancy_streak"] = out["upper_zone"].astype(int).groupby((~out["upper_zone"]).cumsum()).cumsum()
        out["eligible"] = out["occupancy_streak"].ge(params.min_streak)
        out["confirmed"] = out["eligible"] & out["eligible"].shift(1, fill_value=False)
        return out

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: PositionStreakParams) -> SignalFrame:
        df = pd.DataFrame(index=data.index)
        # Within an uninterrupted eligible streak, confirmation stays true.
        # Losing eligibility therefore exits immediately at the next fill.
        df["signal"] = indicators["confirmed"].fillna(False).astype(int)
        df["size"] = 0.95
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check() -> None:
    index = pd.date_range("2020-01-01", periods=17, freq="D")
    closes = [2.0] * 8 + [8.0] * 4 + [2.0] + [8.0] * 4
    data = pd.DataFrame({"open": closes, "high": 10.0, "low": 0.0, "close": closes, "volume": 1000.0}, index=index)
    params = PositionStreakParams()
    strategy = GeneratedStrategy()
    derived = strategy.indicators(data, params)
    result = strategy.generate_signals(data, derived, None, params).data
    assert result["signal"].tolist() == [0] * 11 + [1, 1, 0, 0, 0, 1]
    assert result["size"].gt(0.0).all()
    flat = data.assign(open=5.0, high=5.0, low=5.0, close=5.0)
    flat_result = strategy.generate_signals(flat, strategy.indicators(flat, params), None, params).data
    assert flat_result["signal"].eq(0).all()


if __name__ == "__main__":
    _self_check()
