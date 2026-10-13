from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class AnalogParams:
    history: int = 126
    atr_period: int = 14
    radius: float = 0.65
    min_matches: int = 8
    min_edge: float = 0.002
    stop_atr: float = 2.0
    hold_bars: int = 4


class GeneratedStrategy(BaseStrategy[AnalogParams]):
    strategy_id = "gen_local_1790919216"

    @classmethod
    def params_type(cls):
        return AnalogParams

    @staticmethod
    def warmup_bars(params: AnalogParams) -> int:
        return int(max(1, params.history) + max(1, params.hold_bars) + max(1, params.atr_period) + 3)

    def indicators(self, data: pd.DataFrame, params: AnalogParams) -> pd.DataFrame:
        out = pd.DataFrame(index=data.index)
        previous = data["close"].shift(1)
        true_range = pd.concat([
            data["high"] - data["low"],
            (data["high"] - previous).abs(),
            (data["low"] - previous).abs(),
        ], axis=1).max(axis=1)
        out["atr"] = true_range.rolling(max(1, params.atr_period)).mean()
        denominator = out["atr"].shift(1).where(out["atr"].shift(1) > 0)
        out["gap"] = ((data["open"] - previous) / denominator).replace([np.inf, -np.inf], np.nan)
        out["gap_1"] = out["gap"].shift(1)
        out["gap_2"] = out["gap"].shift(2)
        count = pd.Series(0.0, index=data.index)
        total = pd.Series(0.0, index=data.index)
        horizon = max(1, params.hold_bars)
        radius_squared = max(0.0, params.radius) ** 2
        for lag in range(horizon + 1, horizon + max(1, params.history) + 1):
            distance = (
                (out["gap"] - out["gap"].shift(lag)) ** 2
                + (out["gap_1"] - out["gap_1"].shift(lag)) ** 2
                + (out["gap_2"] - out["gap_2"].shift(lag)) ** 2
            ) / 3.0
            origin = data["close"].shift(lag).where(data["close"].shift(lag) > 0)
            outcome = (data["close"].shift(lag - horizon) / origin - 1.0).replace([np.inf, -np.inf], np.nan)
            matched = distance.le(radius_squared) & outcome.notna()
            count += matched.astype(float)
            total += outcome.where(matched, 0.0)
        out["matches"] = count
        out["analog_return"] = total.div(count.where(count > 0))
        return out

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame,
                         ctx: StrategyContext, params: AnalogParams) -> SignalFrame:
        eligible = (
            indicators["analog_return"].gt(params.min_edge)
            & indicators["matches"].ge(max(1, params.min_matches))
            & indicators["gap"].lt(0)
            & data["close"].gt(data["open"])
            & indicators["atr"].gt(0)
        ).fillna(False)
        confirmed = eligible & eligible.shift(1, fill_value=False)
        signals = np.zeros(len(data), dtype=int)
        closes = data["close"].to_numpy(dtype=float)
        atrs = indicators["atr"].to_numpy(dtype=float)
        entries = confirmed.to_numpy(dtype=bool)
        active = False
        age = 0
        highest = -np.inf
        stop = -np.inf
        for i in range(len(data)):
            if active:
                age += 1
                if np.isfinite(closes[i]):
                    highest = max(highest, closes[i])
                    if np.isfinite(atrs[i]) and atrs[i] > 0:
                        stop = max(stop, highest - max(0.0, params.stop_atr) * atrs[i])
                stopped = np.isfinite(closes[i]) and closes[i] <= stop
                if stopped or age >= max(1, params.hold_bars):
                    active = False
                else:
                    signals[i] = 1
            elif entries[i] and np.isfinite(closes[i]):
                active = True
                age = 0
                highest = -np.inf
                stop = -np.inf
                signals[i] = 1
        df = pd.DataFrame({"signal": signals, "size": 1.0}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


if __name__ == "__main__":
    index = pd.date_range("2020-01-01", periods=8)
    close = pd.Series([100., 101., 102., 100., 102., 99., 100., 101.], index=index)
    data = pd.DataFrame({"open": close - 0.5, "high": close + 1.0,
                         "low": close - 1.0, "close": close, "volume": 1000.0})
    derived = pd.DataFrame({"atr": 1.0, "gap": -0.2, "matches": 10.0,
                            "analog_return": [0., 0.01, 0.01, 0., 0., 0., 0., 0.]}, index=index)
    result = GeneratedStrategy.generate_signals(None, data, derived, None, AnalogParams())
    assert result.data["signal"].tolist() == [0, 0, 0, 1, 1, 1, 0, 0]
    assert result.data["size"].gt(0).all()
