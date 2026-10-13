from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class ShapeParams:
    peak_window: int = 20
    atr_window: int = 14
    analog_window: int = 180
    min_depth: float = 0.8
    min_forecast: float = 0.001
    trail_atr: float = 1.2
    hold_days: int = 2


class GeneratedStrategy(BaseStrategy[ShapeParams]):
    strategy_id = "gen_local-w2_1791330139"

    @classmethod
    def params_type(cls) -> type[ShapeParams]:
        return ShapeParams

    def warmup_bars(self, params: ShapeParams) -> int:
        return int(max(params.peak_window, params.atr_window + 1) + params.analog_window + 4)

    def indicators(self, data: pd.DataFrame, params: ShapeParams) -> pd.DataFrame:
        if (params.peak_window < 2 or params.atr_window < 2
                or params.analog_window < 8 or params.min_depth < 0
                or params.min_forecast < 0 or params.trail_atr <= 0
                or params.hold_days not in (1, 2)
                or not all(np.isfinite(x) for x in
                           (params.min_depth, params.min_forecast, params.trail_atr))):
            raise ValueError("Invalid strategy parameters")
        out = pd.DataFrame(index=data.index)
        close = data["close"].where(data["close"] > 0)
        previous = close.shift(1)
        true_range = pd.concat([
            data["high"] - data["low"],
            (data["high"] - previous).abs(),
            (data["low"] - previous).abs(),
        ], axis=1).max(axis=1)
        out["true_range"] = true_range
        out["atr"] = true_range.rolling(params.atr_window).mean().where(lambda x: x > 0)
        out["peak"] = close.rolling(params.peak_window).max()
        out["depth"] = (out["peak"] - close) / out["atr"]
        out["depth_1"] = out["depth"].shift(1)
        out["depth_2"] = out["depth"].shift(2)
        span = (data["high"] - data["low"]).where(lambda x: x > 0)
        out["close_location"] = (close - data["low"]) / span
        out["close_location_1"] = out["close_location"].shift(1)
        out["two_day_return"] = close / close.shift(2) - 1.0

        shape = out[["depth", "depth_1", "depth_2"]]
        distances = []
        outcomes = []
        # Only analogs whose two-day outcomes were known before today's close.
        for lag in range(3, params.analog_window + 3):
            distance = ((shape - shape.shift(lag)) ** 2).mean(axis=1, skipna=False)
            outcome = out["two_day_return"].shift(lag - 2)
            valid = np.isfinite(distance) & np.isfinite(outcome)
            distances.append(distance.where(valid, np.inf).to_numpy())
            outcomes.append(outcome.to_numpy())
        distance_matrix = np.column_stack(distances)
        outcome_matrix = np.column_stack(outcomes)
        nearest = np.argsort(distance_matrix, axis=1)[:, :8]
        matched_distance = np.take_along_axis(distance_matrix, nearest, axis=1)
        matched_return = np.take_along_axis(outcome_matrix, nearest, axis=1)
        usable = np.isfinite(matched_distance) & (matched_distance <= 0.75 ** 2)
        counts = usable.sum(axis=1)
        out["analog_count"] = counts
        out["analog_forecast"] = np.where(
            counts >= 5,
            np.where(usable, matched_return, 0.0).sum(axis=1) / np.maximum(counts, 1),
            np.nan,
        )
        return out.replace([np.inf, -np.inf], np.nan)

    def generate_signals(
        self, data: pd.DataFrame, indicators: pd.DataFrame,
        ctx: StrategyContext, params: ShapeParams,
    ) -> SignalFrame:
        depth_agrees = (
            (indicators["depth"] >= params.min_depth)
            & (indicators["analog_count"] >= 5)
            & (indicators["analog_forecast"] > params.min_forecast)
        )
        range_agrees = (
            (indicators["close_location"] >= 0.65)
            & (indicators["close_location"] - indicators["close_location_1"] >= 0.20)
        )
        entries = (depth_agrees & range_agrees).fillna(False).to_numpy()
        closes = data["close"].to_numpy(dtype=float)
        atrs = indicators["atr"].to_numpy(dtype=float)
        decisions = np.zeros(len(data), dtype=int)
        active = False
        entry_bar = -1
        high_water = -np.inf
        stop = -np.inf
        for i in range(len(data)):
            valid = np.isfinite(closes[i]) and closes[i] > 0 and np.isfinite(atrs[i]) and atrs[i] > 0
            if active:
                if not valid:
                    active = False
                    continue
                # The first in-trade close is on the bar after the entry decision.
                high_water = max(high_water, closes[i])
                stop = max(stop, high_water - params.trail_atr * atrs[i])
                if closes[i] <= stop or i - entry_bar + 1 >= params.hold_days:
                    active = False
                else:
                    decisions[i] = 1
            elif valid and entries[i]:
                active = True
                entry_bar = i + 1
                high_water = -np.inf
                stop = -np.inf
                decisions[i] = 1
        df = pd.DataFrame({"signal": decisions, "size": 0.95}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check() -> None:
    params = ShapeParams()
    strategy = GeneratedStrategy()
    index = pd.date_range("2020-01-01", periods=6, freq="D")
    data = pd.DataFrame({
        "open": [100., 100., 102., 98., 100., 100.],
        "high": [101., 101., 103., 99., 101., 101.],
        "low": [99., 99., 101., 97., 99., 99.],
        "close": [100., 100., 102., 98., 100., 100.],
        "volume": [1000.] * 6,
    }, index=index)
    features = pd.DataFrame({
        "atr": [2., 2., 2., 2., 2., 2.],
        "depth": [1.] * 6,
        "analog_count": [8] * 6,
        "analog_forecast": [0., 0.01, 0., 0., 0., 0.],
        "close_location": [0.3, 0.8, 0.3, 0.3, 0.3, 0.3],
        "close_location_1": [0.3] * 6,
    }, index=index)
    result = strategy.generate_signals(data, features, None, params).data
    assert result["signal"].tolist() == [0, 0, 1, 1, 0, 0]
    features["close_location"] = 0.3
    assert not strategy.generate_signals(data, features, None, params).data["signal"].any()
    assert strategy.indicators(data.iloc[:0], params).empty


if __name__ == "__main__":
    _self_check()
