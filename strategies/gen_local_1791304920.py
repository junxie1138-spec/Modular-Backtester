from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class PositionStreakParams:
    range_bars: int = 3
    entry_streak: int = 2
    edge: float = 0.65
    streak_cap: int = 4
    min_size: float = 0.20
    max_size: float = 0.95


class GeneratedStrategy(BaseStrategy[PositionStreakParams]):
    strategy_id = "gen_local_1791304920"

    @classmethod
    def params_type(cls) -> type[PositionStreakParams]:
        return PositionStreakParams

    def warmup_bars(self, params: PositionStreakParams) -> int:
        return int(params.range_bars + params.streak_cap + 1)

    def indicators(self, data: pd.DataFrame, params: PositionStreakParams) -> pd.DataFrame:
        if not (
            params.range_bars >= 1
            and 1 <= params.entry_streak <= params.streak_cap
            and 0.5 < params.edge < 1.0
            and 0.0 < params.min_size <= params.max_size <= 1.0
        ):
            raise ValueError("Invalid position-streak parameters")

        out = pd.DataFrame(index=data.index)
        out["range_low"] = data["low"].rolling(params.range_bars).min().shift(1)
        out["range_high"] = data["high"].rolling(params.range_bars).max().shift(1)
        out["range_width"] = out["range_high"] - out["range_low"]
        out["position"] = (
            (data["close"] - out["range_low"])
            / out["range_width"].where(out["range_width"] > 0.0)
        ).replace([np.inf, -np.inf], np.nan)
        out["position_change"] = out["position"].diff()
        out["valid"] = out["position"].notna() & out["position_change"].notna()

        rising = out["valid"] & out["position_change"].gt(0.0)
        falling = out["valid"] & out["position_change"].lt(0.0)
        up_prefix = pd.Series(True, index=data.index)
        down_prefix = pd.Series(True, index=data.index)
        up_count = pd.Series(0, index=data.index, dtype=int)
        down_count = pd.Series(0, index=data.index, dtype=int)
        for lag in range(params.streak_cap):
            up_prefix = up_prefix & rising.shift(lag, fill_value=False)
            down_prefix = down_prefix & falling.shift(lag, fill_value=False)
            up_count = up_count + up_prefix.astype(int)
            down_count = down_count + down_prefix.astype(int)

        out["up_streak"] = up_count
        out["down_streak"] = down_count
        out["stretch"] = (2.0 * (out["position"] - 0.5).abs()).clip(0.0, 1.0)
        out["strength"] = (
            out[["up_streak", "down_streak"]].max(axis=1)
            / float(params.streak_cap)
            * out["stretch"]
        ).fillna(0.0).clip(0.0, 1.0)
        return out.fillna(0.0)

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: PositionStreakParams,
    ) -> SignalFrame:
        long_entry = (
            indicators["valid"].astype(bool)
            & indicators["down_streak"].ge(params.entry_streak)
            & indicators["position"].le(1.0 - params.edge)
        )
        short_entry = (
            indicators["valid"].astype(bool)
            & indicators["up_streak"].ge(params.entry_streak)
            & indicators["position"].ge(params.edge)
        )
        intent = np.select([long_entry, short_entry], [1, -1], default=0)
        entry_sizes = (
            params.min_size
            + (params.max_size - params.min_size) * indicators["strength"]
        ).to_numpy(dtype=float)

        signals = np.zeros(len(data), dtype=int)
        sizes = np.full(len(data), params.min_size, dtype=float)
        position = 0
        position_size = params.min_size
        for i, desired in enumerate(intent):
            if position != 0:
                if desired == -position:
                    position = 0
            elif desired != 0:
                position = int(desired)
                position_size = float(entry_sizes[i])
            signals[i] = position
            sizes[i] = position_size

        df = pd.DataFrame({"signal": signals, "size": sizes}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = df["size"].shift(1).fillna(params.min_size).astype(float)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check() -> None:
    strategy = object.__new__(GeneratedStrategy)
    params = PositionStreakParams()
    index = pd.date_range("2020-01-01", periods=5)
    data = pd.DataFrame(
        {"open": 100.0, "high": 101.0, "low": 99.0, "close": 100.0, "volume": 1000.0},
        index=index,
    )
    indicators = pd.DataFrame(
        {
            "valid": [True] * 5,
            "down_streak": [2, 0, 0, 0, 2],
            "up_streak": [0, 0, 2, 3, 0],
            "position": [0.2, 0.5, 0.8, 0.9, 0.2],
            "strength": [0.3, 0.0, 0.3, 0.6, 0.3],
        },
        index=index,
    )
    result = strategy.generate_signals(data, indicators, None, params).data
    assert result["signal"].tolist() == [0, 1, 1, 0, -1]
    assert np.isclose(result["size"].iloc[1], 0.425)
    assert np.isclose(result["size"].iloc[4], 0.65)
    assert result["size"].gt(0.0).all()
    computed = strategy.indicators(data, params)
    assert computed.index.equals(data.index)
    assert not computed.isna().any().any()
    assert strategy.warmup_bars(params) == 8


if __name__ == "__main__":
    _self_check()
