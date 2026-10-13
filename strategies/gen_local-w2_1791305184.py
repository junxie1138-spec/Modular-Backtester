from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class StreakAnalogParams:
    minimum_edge: float = 0.002
    profit_target: float = 0.015


class GeneratedStrategy(BaseStrategy[StreakAnalogParams]):
    strategy_id = "gen_local-w2_1791305184"

    @classmethod
    def params_type(cls) -> type[StreakAnalogParams]:
        return StreakAnalogParams

    @staticmethod
    def warmup_bars(params: StreakAnalogParams) -> int:
        return 518

    @staticmethod
    def indicators(data: pd.DataFrame, params: StreakAnalogParams) -> pd.DataFrame:
        close = data["close"].astype(float)
        change = close.diff()
        direction = np.sign(change).fillna(0)
        length = pd.Series(1, index=data.index, dtype=int)
        continuing = direction.ne(0)
        # Six is the fixed saturation ceiling for the streak representation.
        for lag in range(1, 6):
            continuing = continuing & direction.eq(direction.shift(lag))
            length = length + continuing.astype(int)
        streak = direction * length
        previous_streak = streak.shift(3).clip(-3, 3)
        shape = (streak + 6) * 7 + previous_streak + 3
        historical_shape = shape.shift(4)
        entry_open = data["open"].astype(float).shift(3)
        realized_return = (close / entry_open.where(entry_open.gt(0)) - 1).replace([np.inf, -np.inf], np.nan)
        valid_outcome = realized_return.notna() & historical_shape.notna()
        forecast = pd.Series(np.nan, index=data.index, dtype=float)
        analog_count = pd.Series(0.0, index=data.index)
        for state in range(91):
            matched = historical_shape.eq(state) & valid_outcome
            count = matched.astype(float).rolling(504, min_periods=504).sum()
            total = realized_return.where(matched, 0.0).rolling(504, min_periods=504).sum()
            mean = total / count.where(count.ge(6))
            current = shape.eq(state)
            forecast.loc[current] = mean.loc[current]
            analog_count.loc[current] = count.loc[current]
        ready = pd.Series(np.arange(len(data)) >= 517, index=data.index)
        forecast = forecast.where(ready).replace([np.inf, -np.inf], np.nan)
        return pd.DataFrame({
            "direction": direction,
            "streak": streak,
            "previous_streak": previous_streak,
            "shape": shape,
            "historical_shape": historical_shape,
            "realized_return": realized_return,
            "analog_count": analog_count,
            "forecast": forecast,
        }, index=data.index)

    @staticmethod
    def generate_signals(
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: StreakAnalogParams,
    ) -> SignalFrame:
        if not np.isfinite(params.minimum_edge) or params.minimum_edge <= 0:
            raise ValueError("minimum_edge must be finite and positive")
        if not np.isfinite(params.profit_target) or params.profit_target <= 0:
            raise ValueError("profit_target must be finite and positive")
        forecast = indicators["forecast"].to_numpy(dtype=float)
        opens = data["open"].to_numpy(dtype=float)
        closes = data["close"].to_numpy(dtype=float)
        signals = np.zeros(len(data), dtype=int)
        position = 0
        entry_price = np.nan
        age = 0
        for i in range(len(data)):
            if position != 0:
                # Yesterday's decision becomes an exposure at today's open.
                if age == 0:
                    entry_price = opens[i]
                age += 1
                valid_entry = np.isfinite(entry_price) and entry_price > 0
                gain = position * (closes[i] / entry_price - 1) if valid_entry else np.nan
                if not valid_entry or age >= 4 or (np.isfinite(gain) and gain >= params.profit_target):
                    position = 0
                    entry_price = np.nan
                    age = 0
            elif np.isfinite(closes[i]) and closes[i] > 0 and np.isfinite(forecast[i]):
                if forecast[i] >= params.minimum_edge:
                    position = 1
                elif forecast[i] <= -params.minimum_edge:
                    position = -1
            signals[i] = position
        df = pd.DataFrame({"signal": signals, "size": 1.0}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check() -> None:
    index = pd.date_range("2020-01-01", periods=12, freq="D")
    data = pd.DataFrame({
        "open": 100.0,
        "high": 103.0,
        "low": 99.0,
        "close": [100, 100, 102, 100, 100, 100, 100, 100, 100, 100, 100, 100],
        "volume": 1000.0,
    }, index=index)
    indicators = pd.DataFrame({"forecast": [0.01, 0, 0, 0, 0, 0, -0.01, 0, 0, 0, 0, 0]}, index=index)
    result = GeneratedStrategy.generate_signals(data, indicators, None, StreakAnalogParams()).data
    # Long profit exit, short four-session exit, and next-bar execution.
    assert result["signal"].tolist() == [0, 1, 1, 0, 0, 0, 0, -1, -1, -1, -1, 0]
    extended = pd.concat([data] * 50, ignore_index=True)
    extended.index = pd.date_range("2020-01-01", periods=len(extended), freq="D")
    full = GeneratedStrategy.indicators(extended, StreakAnalogParams())
    cutoff = 550
    prefix = GeneratedStrategy.indicators(extended.iloc[:cutoff], StreakAnalogParams())
    pd.testing.assert_frame_equal(full.iloc[:cutoff], prefix)


if __name__ == "__main__":
    _self_check()
