from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class CalendarStreakParams:
    streak_length: int = 2
    holding_bars: int = 18
    atr_period: int = 14
    atr_multiple: float = 3.0
    early_month_sessions: int = 5
    short_start_day: int = 15
    short_end_day: int = 21


class GeneratedStrategy(BaseStrategy[CalendarStreakParams]):
    strategy_id = "gen_local-w2_1791220356"

    @classmethod
    def params_type(cls) -> type[CalendarStreakParams]:
        return CalendarStreakParams

    def warmup_bars(self, params: CalendarStreakParams) -> int:
        return int(max(200, params.atr_period + 1, params.streak_length + 1))

    def indicators(self, data: pd.DataFrame, params: CalendarStreakParams) -> pd.DataFrame:
        if min(params.streak_length, params.holding_bars, params.atr_period, params.early_month_sessions) < 1:
            raise ValueError("Window lengths must be positive")
        if not np.isfinite(params.atr_multiple) or params.atr_multiple <= 0:
            raise ValueError("atr_multiple must be finite and positive")
        if not 1 <= params.short_start_day <= params.short_end_day <= 31:
            raise ValueError("Invalid short calendar window")
        result = pd.DataFrame(index=data.index)
        close = data["close"].astype(float)
        previous = close.shift(1)
        change = close.diff()
        result["change"] = change
        result["ma200"] = close.rolling(200, min_periods=200).mean()
        true_range = pd.concat([
            data["high"] - data["low"],
            (data["high"] - previous).abs(),
            (data["low"] - previous).abs(),
        ], axis=1).max(axis=1)
        valid_range = previous.notna() & data[["high", "low", "close"]].notna().all(axis=1)
        result["true_range"] = true_range.where(valid_range)
        result["atr"] = result["true_range"].rolling(params.atr_period, min_periods=params.atr_period).mean()
        down = change.lt(0)
        up = change.gt(0)
        result["down_streak"] = down.astype(int).groupby((~down).cumsum()).cumsum()
        result["up_streak"] = up.astype(int).groupby((~up).cumsum()).cumsum()
        months = data.index.to_period("M")
        result["month_session"] = pd.Series(1, index=data.index).groupby(months).cumsum()
        result["calendar_day"] = data.index.day
        ready = np.isfinite(close) & np.isfinite(result["ma200"]) & np.isfinite(result["atr"]) & result["atr"].gt(0)
        result["long_setup"] = (
            ready & close.gt(result["ma200"])
            & result["month_session"].le(params.early_month_sessions)
            & result["down_streak"].ge(params.streak_length)
        ).fillna(False)
        result["short_setup"] = (
            ready & close.lt(result["ma200"])
            & result["calendar_day"].between(params.short_start_day, params.short_end_day)
            & result["up_streak"].ge(params.streak_length)
        ).fillna(False)
        return result

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: CalendarStreakParams) -> SignalFrame:
        closes = data["close"].to_numpy(dtype=float)
        atrs = indicators["atr"].to_numpy(dtype=float)
        averages = indicators["ma200"].to_numpy(dtype=float)
        longs = indicators["long_setup"].fillna(False).to_numpy(dtype=bool)
        shorts = indicators["short_setup"].fillna(False).to_numpy(dtype=bool)
        signals = np.zeros(len(data), dtype=int)
        position = 0
        age = 0
        high_water = np.nan
        stop = np.nan
        for i in range(len(data)):
            close = closes[i]
            atr = atrs[i]
            average = averages[i]
            valid = np.isfinite(close) and np.isfinite(atr) and atr > 0 and np.isfinite(average)
            if position:
                age += 1
                if not valid:
                    position = 0
                else:
                    # Initialize from the first held bar, after next-bar execution.
                    high_water = close if np.isnan(high_water) else max(high_water, close)
                    candidate = high_water - params.atr_multiple * atr
                    stop = candidate if np.isnan(stop) else max(stop, candidate)
                    regime_lost = (position == 1 and close <= average) or (position == -1 and close >= average)
                    # The required highest-close exit applies to both directions;
                    # for shorts it realizes a sufficiently large downward move.
                    if close <= stop or age >= params.holding_bars or regime_lost:
                        position = 0
            elif valid:
                if longs[i]:
                    position = 1
                elif shorts[i]:
                    position = -1
                if position:
                    age = 0
                    high_water = np.nan
                    stop = np.nan
            signals[i] = position
        df = pd.DataFrame(index=data.index)
        df["signal"] = signals
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = 1.0
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check() -> None:
    index = pd.date_range("2020-01-01", periods=6)
    closes = np.array([100.0, 102.0, 104.0, 100.0, 100.0, 100.0])
    data = pd.DataFrame({"open": closes, "high": closes + 1, "low": closes - 1, "close": closes, "volume": 1000.0}, index=index)
    features = pd.DataFrame({"atr": 1.0, "ma200": 90.0, "long_setup": [True, False, False, False, False, False], "short_setup": False}, index=index)
    strategy = GeneratedStrategy()
    params = CalendarStreakParams()
    result = strategy.generate_signals(data, features, None, params).data
    assert result["signal"].tolist() == [0, 1, 1, 1, 0, 0]
    features["atr"] = [1.0, 1.0, 10.0, 10.0, 10.0, 10.0]
    data.loc[index[3], "close"] = 98.0
    result = strategy.generate_signals(data, features, None, params).data
    assert result["signal"].iloc[4] == 0
    assert result["size"].gt(0).all()


if __name__ == "__main__":
    _self_check()
