from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class VolatilityMemoryParams:
    lookback: int = 24
    memory_threshold: float = 0.15
    impulse_floor: float = 0.7
    hold_bars: int = 4


class GeneratedStrategy(BaseStrategy[VolatilityMemoryParams]):
    strategy_id = "gen_local_1791196909"

    @classmethod
    def params_type(cls) -> type[VolatilityMemoryParams]:
        return VolatilityMemoryParams

    @staticmethod
    def warmup_bars(params: VolatilityMemoryParams) -> int:
        return int(params.lookback) + 3

    @staticmethod
    def indicators(data: pd.DataFrame, params: VolatilityMemoryParams) -> pd.DataFrame:
        if params.lookback < 3 or not 3 <= params.hold_bars <= 5:
            raise ValueError("lookback must be >= 3 and hold_bars must be 3 through 5")
        if not np.isfinite(params.memory_threshold) or not 0 < params.memory_threshold < 1:
            raise ValueError("memory_threshold must be finite and between 0 and 1")
        if not np.isfinite(params.impulse_floor) or params.impulse_floor < 0:
            raise ValueError("impulse_floor must be finite and nonnegative")
        out = pd.DataFrame(index=data.index)
        close = data["close"].where(data["close"] > 0)
        out["return"] = np.log(close).diff().replace([np.inf, -np.inf], np.nan)
        out["pair_return"] = out["return"] + out["return"].shift(1)
        out["daily_std"] = out["return"].rolling(params.lookback, min_periods=params.lookback).std(ddof=0)
        out["pair_std"] = out["pair_return"].rolling(params.lookback, min_periods=params.lookback).std(ddof=0)
        variance = out["daily_std"].pow(2).replace(0.0, np.nan)
        out["memory"] = (out["pair_std"].pow(2) / (2.0 * variance) - 1.0).clip(-1.0, 1.0)
        out["impulse"] = out["return"] / out["daily_std"].shift(1).replace(0.0, np.nan)
        valid = (
            np.isfinite(out["memory"])
            & np.isfinite(out["impulse"])
            & (out["memory"].abs() >= params.memory_threshold)
            & (out["impulse"].abs() >= params.impulse_floor)
        )
        forecast = np.sign(out["return"]) * np.sign(out["memory"])
        out["candidate"] = forecast.where(valid, 0).fillna(0).astype(int)
        consecutive = out["candidate"].eq(out["candidate"].shift(1)) & out["candidate"].ne(0)
        out["confirmed"] = out["candidate"].where(consecutive, 0).astype(int)
        return out

    @staticmethod
    def generate_signals(
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: VolatilityMemoryParams,
    ) -> SignalFrame:
        if not 3 <= params.hold_bars <= 5:
            raise ValueError("hold_bars must be 3 through 5")
        candidates = indicators["confirmed"].fillna(0).to_numpy(dtype=int)
        signals = np.zeros(len(data), dtype=int)
        position = 0
        exit_bar = -1
        for i, candidate in enumerate(candidates):
            if position != 0:
                if i == exit_bar:
                    position = 0
                    continue
                signals[i] = position
            elif candidate != 0:
                position = int(candidate)
                exit_bar = i + params.hold_bars
                signals[i] = position
        df = pd.DataFrame({"signal": signals, "size": 1.0}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


if __name__ == "__main__":
    params = VolatilityMemoryParams()
    index = pd.date_range("2020-01-01", periods=11)
    data = pd.DataFrame(1.0, index=index, columns=["open", "high", "low", "close", "volume"])
    derived = GeneratedStrategy.indicators(data, params)
    assert derived["confirmed"].eq(0).all()
    derived["confirmed"] = [1, 1, -1, -1, -1, -1, 1, 1, 1, -1, -1]
    result = GeneratedStrategy.generate_signals(data, derived, None, params).data
    assert result["signal"].tolist() == [0, 1, 1, 1, 1, 0, -1, -1, -1, -1, 0]
    assert result["size"].gt(0).all()
