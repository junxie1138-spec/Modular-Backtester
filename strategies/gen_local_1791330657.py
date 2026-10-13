from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class QueueParams:
    ma_bars: int = 5
    capacity: float = 3.5
    deadband: float = 0.35


class GeneratedStrategy(BaseStrategy[QueueParams]):
    strategy_id = "gen_local_1791330657"

    @classmethod
    def params_type(cls) -> type[QueueParams]:
        return QueueParams

    def warmup_bars(self, params: QueueParams) -> int:
        if not 2 <= params.ma_bars <= 6:
            raise ValueError("ma_bars must be between 2 and 6")
        if not np.isfinite(params.capacity) or params.capacity <= 0:
            raise ValueError("capacity must be finite and positive")
        if not np.isfinite(params.deadband) or params.deadband < 0:
            raise ValueError("deadband must be finite and nonnegative")
        return int(params.ma_bars + 4)

    def indicators(self, data: pd.DataFrame, params: QueueParams) -> pd.DataFrame:
        GeneratedStrategy.warmup_bars(self, params)
        close = data["close"].astype(float).replace([np.inf, -np.inf], np.nan)
        baseline = close.rolling(params.ma_bars, min_periods=params.ma_bars).mean().shift(1)
        dispersion = close.rolling(params.ma_bars, min_periods=params.ma_bars).std(ddof=0).shift(1)
        z = ((close - baseline) / dispersion.replace(0.0, np.nan)).replace([np.inf, -np.inf], np.nan)
        up_excess = (z - params.deadband).clip(lower=0.0)
        down_excess = (-z - params.deadband).clip(lower=0.0)
        return pd.DataFrame({
            "baseline": baseline,
            "dispersion": dispersion,
            "z": z,
            "up_excess": up_excess,
            "down_excess": down_excess,
            "up_queue": up_excess.rolling(4, min_periods=4).sum(),
            "down_queue": down_excess.rolling(4, min_periods=4).sum(),
        }, index=data.index)

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: QueueParams) -> SignalFrame:
        up = indicators["up_queue"]
        down = indicators["down_queue"]
        z = indicators["z"]
        long_entry = (up >= params.capacity) & (up > down) & (z > params.deadband)
        short_entry = (down >= params.capacity) & (down > up) & (z < -params.deadband)
        direction = np.select([long_entry, short_entry], [1, -1], default=0)
        signals = np.zeros(len(data), dtype=int)
        position = 0
        for i, candidate in enumerate(direction):
            if position == 0:
                position = int(candidate)
            elif candidate == -position:
                position = 0
            signals[i] = position
        df = pd.DataFrame({"signal": signals, "size": 0.95}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


if __name__ == "__main__":
    params = QueueParams()
    index = pd.date_range("2020-01-01", periods=7)
    data = pd.DataFrame({"open": 100.0, "high": 101.0, "low": 99.0, "close": 100.0, "volume": 1000.0}, index=index)
    indicators = pd.DataFrame({
        "z": [np.nan, 2.0, 0.0, -2.0, -2.0, 0.0, 2.0],
        "up_queue": [np.nan, 4.0, 0.0, 0.0, 0.0, 0.0, 4.0],
        "down_queue": [np.nan, 0.0, 0.0, 4.0, 4.0, 0.0, 0.0],
    }, index=index)
    result = GeneratedStrategy.generate_signals(None, data, indicators, None, params)
    assert result.data["signal"].tolist() == [0, 0, 1, 1, 0, -1, -1]
    assert GeneratedStrategy.warmup_bars(None, params) == 9
    flat = GeneratedStrategy.indicators(None, data, params)
    flat_result = GeneratedStrategy.generate_signals(None, data, flat, None, params)
    assert flat_result.data["signal"].eq(0).all()
