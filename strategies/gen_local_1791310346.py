from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class FeedbackParams:
    atr_fast: int = 5
    atr_slow: int = 20
    feedback_window: int = 20
    drift_bars: int = 3
    feedback_min: float = 0.25
    expansion_min: float = 1.05
    trail_atr: float = 0.9
    max_hold: int = 2
    min_size: float = 0.25
    max_size: float = 0.95


class GeneratedStrategy(BaseStrategy[FeedbackParams]):
    strategy_id = "gen_local_1791310346"

    @classmethod
    def params_type(cls) -> type[FeedbackParams]:
        return FeedbackParams

    @staticmethod
    def warmup_bars(params: FeedbackParams) -> int:
        return int(max(params.atr_slow, params.atr_fast + params.feedback_window + 1, params.feedback_window + 2, params.drift_bars + 1))

    @staticmethod
    def indicators(data: pd.DataFrame, params: FeedbackParams) -> pd.DataFrame:
        if min(params.atr_fast, params.atr_slow, params.feedback_window, params.drift_bars) < 2:
            raise ValueError("Lookbacks must be at least two bars")
        if params.max_hold not in (1, 2) or params.trail_atr <= 0:
            raise ValueError("Holding limit must be one or two bars and trail ATR must be positive")
        if not (0 < params.feedback_min < 1 and params.expansion_min > 0 and 0 < params.min_size <= params.max_size <= 1):
            raise ValueError("Invalid feedback, expansion, or sizing parameters")
        out = pd.DataFrame(index=data.index)
        close = data["close"]
        out["previous_close"] = close.shift(1)
        out["true_range"] = pd.concat([
            data["high"] - data["low"],
            (data["high"] - out["previous_close"]).abs(),
            (data["low"] - out["previous_close"]).abs(),
        ], axis=1).max(axis=1)
        out["atr_fast"] = out["true_range"].rolling(params.atr_fast, min_periods=params.atr_fast).mean()
        out["atr_slow"] = out["true_range"].rolling(params.atr_slow, min_periods=params.atr_slow).mean()
        out["return"] = close.div(close.shift(1).replace(0, np.nan)).sub(1)
        out["lagged_return"] = out["return"].shift(1)
        out["atr_growth"] = out["atr_fast"].div(out["atr_fast"].shift(1).replace(0, np.nan)).sub(1)
        out["feedback"] = out["lagged_return"].rolling(params.feedback_window, min_periods=params.feedback_window).corr(out["atr_growth"]).clip(-1, 1)
        out["expansion"] = out["atr_fast"].div(out["atr_slow"].replace(0, np.nan))
        out["drift"] = close.sub(close.shift(params.drift_bars)).div(out["atr_fast"].replace(0, np.nan))
        out = out.replace([np.inf, -np.inf], np.nan)
        out["ready"] = out[["atr_fast", "feedback", "expansion", "drift"]].notna().all(axis=1) & out["atr_fast"].gt(0)
        out["strength"] = (out["feedback"].abs() * out["drift"].abs().clip(upper=2) / 2).clip(0, 1).fillna(0)
        out["position_size"] = params.min_size + (params.max_size - params.min_size) * out["strength"]
        out["entry_direction"] = np.where(
            out["ready"] & out["feedback"].abs().ge(params.feedback_min)
            & out["expansion"].ge(params.expansion_min)
            & out["atr_growth"].gt(0)
            & (out["feedback"] * out["drift"]).gt(0),
            np.sign(out["drift"].fillna(0)), 0,
        ).astype(int)
        return out

    @staticmethod
    def generate_signals(data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: FeedbackParams) -> SignalFrame:
        count = len(data)
        signals = np.zeros(count, dtype=int)
        sizes = np.full(count, params.min_size, dtype=float)
        closes = data["close"].to_numpy(dtype=float)
        atrs = indicators["atr_fast"].to_numpy(dtype=float)
        entries = indicators["entry_direction"].to_numpy(dtype=int)
        convictions = indicators["position_size"].to_numpy(dtype=float)
        position = 0
        age = 0
        high_water = low_water = stop = 0.0
        entry_size = params.min_size
        for i in range(count):
            close, atr = closes[i], atrs[i]
            if not (np.isfinite(close) and np.isfinite(atr) and atr > 0):
                position = 0
                age = 0
                continue
            if position:
                age += 1
                high_water = max(high_water, close)
                low_water = min(low_water, close)
                if position == 1:
                    stop = max(stop, high_water - params.trail_atr * atr)
                    stopped = close <= stop
                else:
                    stop = min(stop, low_water + params.trail_atr * atr)
                    stopped = close >= stop
                if stopped or age >= params.max_hold:
                    position = 0
            elif entries[i]:
                position = int(entries[i])
                age = 0
                high_water = low_water = close
                stop = close - position * params.trail_atr * atr
                entry_size = float(convictions[i])
            signals[i] = position
            if position:
                sizes[i] = entry_size
        df = pd.DataFrame({"signal": signals, "size": sizes}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = df["size"].shift(1).fillna(params.min_size).astype(float)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check() -> None:
    params = FeedbackParams()
    index = pd.date_range("2020-01-01", periods=4)
    for direction in (1, -1):
        data = pd.DataFrame({"close": [100.0, 100.0 - 2 * direction, 100.0, 100.0]}, index=index)
        indicators = pd.DataFrame({"atr_fast": 1.0, "entry_direction": [direction, 0, 0, 0], "position_size": 0.7}, index=index)
        result = GeneratedStrategy.generate_signals(data, indicators, None, params).data
        assert result["signal"].tolist() == [0, direction, 0, 0]
        assert result["size"].iloc[1] == 0.7
        data["close"] = [100.0, 100.0 + direction, 100.0 + 2 * direction, 100.0 + 3 * direction]
        result = GeneratedStrategy.generate_signals(data, indicators, None, params).data
        assert result["signal"].tolist() == [0, direction, direction, 0]


if __name__ == "__main__":
    _self_check()
