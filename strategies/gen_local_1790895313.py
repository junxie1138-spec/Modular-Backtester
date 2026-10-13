from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class Params:
    memory: int = 18
    compression: float = 0.75


class GeneratedStrategy(BaseStrategy[Params]):
    strategy_id = "gen_local_1790895313"

    @classmethod
    def params_type(cls) -> type[Params]:
        return Params

    def warmup_bars(self, params: Params) -> int:
        return max(63, 64 + params.memory - 1)

    def indicators(self, data: pd.DataFrame, params: Params) -> pd.DataFrame:
        if params.memory < 2 or not 0.0 < params.compression < 1.0:
            raise ValueError("memory must be >= 2 and compression must be between 0 and 1")
        close = data["close"].astype(float).replace([np.inf, -np.inf], np.nan)
        # ponytail: fixed 64-term fractional filter; extend if truncation materially affects results.
        weights = np.ones(64)
        for k in range(1, 64):
            weights[k] = weights[k - 1] * (k - 1 - 0.4) / k
        fractional_close = close * weights[0]
        for k in range(1, 64):
            fractional_close = fractional_close + close.shift(k) * weights[k]
        window = fractional_close.rolling(params.memory, min_periods=params.memory)
        mean = window.mean()
        median = window.median()
        deviation = window.std(ddof=0).replace(0.0, np.nan)
        tail_balance = (mean - median) / deviation
        daily_range = ((data["high"] - data["low"]) / close.replace(0.0, np.nan)).replace(
            [np.inf, -np.inf], np.nan
        )
        recent_range = daily_range.rolling(5, min_periods=5).mean()
        baseline_range = daily_range.rolling(63, min_periods=63).mean()
        compression_ratio = recent_range / baseline_range.replace(0.0, np.nan)
        return pd.DataFrame({
            "fractional_close": fractional_close,
            "fractional_mean": mean,
            "fractional_median": median,
            "fractional_deviation": deviation,
            "tail_balance": tail_balance,
            "daily_range": daily_range,
            "recent_range": recent_range,
            "baseline_range": baseline_range,
            "compression_ratio": compression_ratio,
        }, index=data.index).replace([np.inf, -np.inf], np.nan)

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: Params,
    ) -> SignalFrame:
        compressed = indicators["compression_ratio"].lt(params.compression)
        balance = indicators["tail_balance"]
        desired = np.select(
            [compressed & balance.gt(0.15), compressed & balance.lt(-0.15)],
            [1, -1],
            default=0,
        )
        signals = np.zeros(len(data), dtype=int)
        position = 0
        for i, direction in enumerate(desired):
            if position == 0:
                position = int(direction)
            elif direction == -position:
                position = 0
            signals[i] = position
        df = pd.DataFrame({"signal": signals, "size": 0.95}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


if __name__ == "__main__":
    strategy = GeneratedStrategy()
    params = Params()
    index = pd.date_range("2020-01-01", periods=240)
    close = 100 + np.arange(240) * 0.04 + np.sin(np.arange(240) / 7)
    data = pd.DataFrame({
        "open": close, "high": close + 0.5, "low": close - 0.5,
        "close": close, "volume": 1000000.0,
    }, index=index)
    derived = strategy.indicators(data, params)
    pd.testing.assert_frame_equal(derived.iloc[:180], strategy.indicators(data.iloc[:180], params))
    frame = pd.DataFrame({
        "compression_ratio": [0.5] * 7,
        "tail_balance": [0.2, 0.2, np.nan, -0.2, -0.2, 0.2, 0.2],
    }, index=index[:7])
    result = strategy.generate_signals(data.iloc[:7], frame, None, params).data
    assert result["signal"].tolist() == [0, 1, 1, 1, 0, -1, 0]
    assert result["size"].gt(0).all()
    flat = data.assign(open=100.0, high=100.0, low=100.0, close=100.0)
    flat_result = strategy.generate_signals(flat, strategy.indicators(flat, params), None, params).data
    assert flat_result["signal"].eq(0).all()
