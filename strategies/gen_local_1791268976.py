from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class RotationParams:
    channel_bars: int = 42
    roc_bars: int = 20
    acceleration_lag: int = 5
    smoothing_bars: int = 5
    noise_bars: int = 20
    acceleration_threshold: float = 0.65


class GeneratedStrategy(BaseStrategy[RotationParams]):
    strategy_id = "gen_local_1791268976"

    @classmethod
    def params_type(cls) -> type[RotationParams]:
        return RotationParams

    def warmup_bars(self, params: RotationParams) -> int:
        return int(max(
            params.channel_bars,
            params.noise_bars + 1,
            params.roc_bars + params.acceleration_lag + params.smoothing_bars,
        ))

    def indicators(self, data: pd.DataFrame, params: RotationParams) -> pd.DataFrame:
        out = pd.DataFrame(index=data.index)
        close = data["close"].replace(0.0, np.nan)
        out["channel_low"] = data["low"].rolling(params.channel_bars).min()
        out["channel_high"] = data["high"].rolling(params.channel_bars).max()
        out["channel_width"] = out["channel_high"] - out["channel_low"]
        out["relative_position"] = (
            (close - out["channel_low"])
            / out["channel_width"].replace(0.0, np.nan)
        )
        out["daily_return"] = close.pct_change(fill_method=None)
        out["roc"] = close.pct_change(params.roc_bars, fill_method=None)
        out["raw_acceleration"] = out["roc"].diff(params.acceleration_lag)
        out["acceleration"] = out["raw_acceleration"].rolling(params.smoothing_bars).mean()
        out["noise"] = out["daily_return"].rolling(params.noise_bars).std(ddof=0)
        out["acceleration_score"] = (
            out["acceleration"]
            / (out["noise"].replace(0.0, np.nan) * np.sqrt(params.acceleration_lag))
        )
        out = out.replace([np.inf, -np.inf], np.nan)
        valid = out[["relative_position", "roc", "acceleration_score"]].notna().all(axis=1)
        long_setup = (
            valid
            & out["relative_position"].lt(0.5)
            & out["roc"].le(0.0)
            & out["acceleration_score"].gt(params.acceleration_threshold)
        )
        short_setup = (
            valid
            & out["relative_position"].gt(0.5)
            & out["roc"].ge(0.0)
            & out["acceleration_score"].lt(-params.acceleration_threshold)
        )
        out["entry_direction"] = np.select([long_setup, short_setup], [1, -1], default=0).astype(int)
        return out

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: RotationParams,
    ) -> SignalFrame:
        setup = indicators["entry_direction"].fillna(0).astype(int)
        confirmed = setup.where(setup.eq(setup.shift(1)), 0).to_numpy()
        positions = np.zeros(len(data), dtype=int)
        position = 0
        for i, direction in enumerate(confirmed):
            if position == 0:
                position = int(direction)
            elif direction == -position:
                position = 0
            positions[i] = position
        df = pd.DataFrame({"signal": positions, "size": 0.95}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


if __name__ == "__main__":
    index = pd.date_range("2020-01-01", periods=9)
    data = pd.DataFrame(index=index)
    indicators = pd.DataFrame({"entry_direction": [1, 1, 0, -1, -1, -1, 0, 1, 1]}, index=index)
    result = GeneratedStrategy().generate_signals(data, indicators, None, RotationParams())
    assert result.data["signal"].tolist() == [0, 0, 1, 1, 1, 0, -1, -1, -1]
    assert result.data["size"].gt(0).all()
