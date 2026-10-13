from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class StreakAnalogParams:
    history_bars: int = 378
    percentile_bars: int = 126
    entry_percentile: float = 0.75
    min_analogs: int = 6
    atr_bars: int = 14
    trail_atr: float = 2.0
    hold_bars: int = 4

    def __post_init__(self):
        if self.history_bars < 40 or self.percentile_bars < 20:
            raise ValueError("History and percentile windows are too short")
        if not 0.0 < self.entry_percentile < 1.0:
            raise ValueError("entry_percentile must be between zero and one")
        if not 1 <= self.min_analogs <= self.history_bars:
            raise ValueError("Invalid minimum analog count")
        if self.atr_bars < 1 or not np.isfinite(self.trail_atr) or self.trail_atr <= 0:
            raise ValueError("ATR settings must be positive")
        if self.hold_bars not in (3, 4, 5):
            raise ValueError("hold_bars must be three, four, or five")


class GeneratedStrategy(BaseStrategy[StreakAnalogParams]):
    strategy_id = "gen_local-w2_1791291774"

    @classmethod
    def params_type(cls):
        return StreakAnalogParams

    def warmup_bars(self, params):
        return int(max(params.atr_bars + 1,
                       params.history_bars + params.percentile_bars
                       + params.hold_bars + 7))

    def indicators(self, data, params):
        close = data["close"].astype(float)
        direction = np.sign(close.diff()).fillna(0.0)
        streak = pd.Series(0.0, index=data.index)
        same = direction.ne(0)
        # Capped counts describe a six-session shape without unbounded history.
        for lag in range(3):
            same = same & direction.eq(direction.shift(lag))
            streak = streak + direction * same.astype(float)
        anchor = streak.shift(3)
        state = (streak + 3) * 7 + anchor + 3
        state = state.where(streak.ne(0) & anchor.notna())
        matured_state = state.shift(params.hold_bars)
        realized = (close / close.shift(params.hold_bars) - 1.0)
        realized = realized.replace([np.inf, -np.inf], np.nan)
        forecast = pd.Series(np.nan, index=data.index, dtype=float)
        analog_count = pd.Series(0.0, index=data.index)

        # Only outcomes completed before the decision bar enter the analog pool.
        for current in (-3, -2, -1, 1, 2, 3):
            for prior in range(-3, 4):
                key = (current + 3) * 7 + prior + 3
                match = matured_state.eq(key) & realized.notna()
                count = match.astype(float).rolling(
                    params.history_bars, min_periods=params.history_bars
                ).sum().shift(1)
                total = realized.where(match, 0.0).rolling(
                    params.history_bars, min_periods=params.history_bars
                ).sum().shift(1)
                estimate = (total / count.replace(0.0, np.nan)).where(
                    count >= params.min_analogs
                )
                selected = state.eq(key)
                forecast = forecast.where(~selected, estimate)
                analog_count = analog_count.where(~selected, count)

        threshold = forecast.abs().rolling(
            params.percentile_bars,
            min_periods=max(10, params.percentile_bars // 2)
        ).quantile(params.entry_percentile).shift(1)
        previous_close = close.shift(1)
        true_range = pd.concat([
            data["high"] - data["low"],
            (data["high"] - previous_close).abs(),
            (data["low"] - previous_close).abs()
        ], axis=1).max(axis=1)
        atr = true_range.rolling(
            params.atr_bars, min_periods=params.atr_bars
        ).mean()
        return pd.DataFrame({
            "direction": direction, "streak": streak, "anchor_streak": anchor,
            "state": state, "matured_state": matured_state,
            "realized_return": realized, "analog_count": analog_count,
            "forecast": forecast, "entry_threshold": threshold,
            "true_range": true_range, "atr": atr
        }, index=data.index)

    def generate_signals(self, data, indicators, ctx, params):
        close = data["close"].to_numpy(dtype=float)
        forecast = indicators["forecast"].to_numpy(dtype=float)
        threshold = indicators["entry_threshold"].to_numpy(dtype=float)
        atr = indicators["atr"].to_numpy(dtype=float)
        signals = np.zeros(len(data), dtype=int)
        position = 0
        entry_bar = -1
        high_water = np.nan
        stop = np.nan

        for i in range(len(data)):
            valid_price = np.isfinite(close[i]) and close[i] > 0
            valid_atr = np.isfinite(atr[i]) and atr[i] > 0
            if position:
                if not valid_price or not valid_atr:
                    position = 0
                else:
                    high_water = max(high_water, close[i])
                    stop = max(stop, high_water - params.trail_atr * atr[i])
                    # The required high-water drawdown exit applies to both sides;
                    # for shorts it harvests a decline from the in-trade high.
                    if close[i] <= stop or i - entry_bar >= params.hold_bars:
                        position = 0
                signals[i] = position
                # No same-close reentry after an exit.
                continue
            if (valid_price and valid_atr and np.isfinite(forecast[i])
                    and np.isfinite(threshold[i])
                    and abs(forecast[i]) > threshold[i]):
                position = 1 if forecast[i] > 0 else -1
                entry_bar = i
                high_water = close[i]
                stop = high_water - params.trail_atr * atr[i]
            signals[i] = position

        df = pd.DataFrame({"signal": signals, "size": 1.0}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check():
    strategy = GeneratedStrategy()
    params = StreakAnalogParams()
    index = pd.date_range("2020-01-01", periods=7, freq="D")
    data = pd.DataFrame({"close": [100., 103., 100., 100., 100., 100., 100.]},
                        index=index)
    indicators = pd.DataFrame({
        "forecast": [1., 0., 0., 0., 0., 0., 0.],
        "entry_threshold": 0.5, "atr": 1.0
    }, index=index)
    result = strategy.generate_signals(data, indicators, None, params).data
    assert result["signal"].tolist() == [0, 1, 1, 0, 0, 0, 0]
    indicators["forecast"] = -indicators["forecast"]
    result = strategy.generate_signals(data, indicators, None, params).data
    assert result["signal"].tolist() == [0, -1, -1, 0, 0, 0, 0]
    data["close"] = 100.0
    result = strategy.generate_signals(data, indicators, None, params).data
    assert result["signal"].tolist() == [0, -1, -1, -1, -1, 0, 0]
    index = pd.date_range("2015-01-01", periods=800, freq="D")
    close = 100.0 * np.exp(np.cumsum(0.002 * np.sin(np.arange(800) * 0.73)))
    data = pd.DataFrame({
        "open": close, "high": close + 1.0, "low": close - 1.0,
        "close": close, "volume": 1000000.0
    }, index=index)
    full = strategy.indicators(data, params)
    prefix = strategy.indicators(data.iloc[:650], params)
    pd.testing.assert_frame_equal(full.iloc[:650], prefix)
    a = strategy.generate_signals(data, full, None, params).data
    b = strategy.generate_signals(data.iloc[:650], prefix, None, params).data
    pd.testing.assert_frame_equal(a.iloc[:650], b)
    assert a["signal"].isin([-1, 0, 1]).all() and a["size"].gt(0).all()


if __name__ == "__main__":
    _self_check()
