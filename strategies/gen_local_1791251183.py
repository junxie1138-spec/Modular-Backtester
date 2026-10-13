from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class GapStepParams:
    survival_bars: int = 4
    trend_span: int = 10
    slope_span: int = 5
    percentile_window: int = 126
    tail_percentile: float = 0.8
    atr_window: int = 14
    trail_atr: float = 3.0
    holding_bars: int = 18


class GeneratedStrategy(BaseStrategy[GapStepParams]):
    strategy_id = "gen_local_1791251183"

    @classmethod
    def params_type(cls):
        return GapStepParams

    def warmup_bars(self, params: GapStepParams) -> int:
        return int(max(
            params.atr_window + 1,
            params.trend_span + params.slope_span
            + params.survival_bars + params.percentile_window + 3,
        ))

    def indicators(self, data: pd.DataFrame, params: GapStepParams) -> pd.DataFrame:
        if min(params.survival_bars, params.trend_span, params.slope_span,
               params.percentile_window, params.atr_window, params.holding_bars) < 1:
            raise ValueError("Lookbacks and holding period must be positive")
        if not 0.5 < params.tail_percentile < 1.0 or params.trail_atr <= 0:
            raise ValueError("Invalid percentile or trailing distance")
        out = pd.DataFrame(index=data.index)
        close = data["close"].where(data["close"] > 0)
        out["previous_close"] = close.shift(1)
        out["log_close"] = np.log(close)
        out["gap"] = np.log(data["open"].where(data["open"] > 0) / out["previous_close"])
        out["gap_magnitude"] = out["gap"].abs()
        out["gap_threshold"] = out["gap_magnitude"].rolling(
            params.percentile_window, min_periods=params.percentile_window
        ).quantile(params.tail_percentile).shift(1)
        out["range"] = data["high"] - data["low"]
        out["high_distance"] = (data["high"] - out["previous_close"]).abs()
        out["low_distance"] = (data["low"] - out["previous_close"]).abs()
        out["true_range"] = out[["range", "high_distance", "low_distance"]].max(axis=1)
        out["atr"] = out["true_range"].rolling(
            params.atr_window, min_periods=params.atr_window
        ).mean()
        out["latent_price"] = out["log_close"].ewm(
            span=params.trend_span, adjust=False, min_periods=params.trend_span
        ).mean()
        out["latent_change"] = out["latent_price"].diff()
        out["latent_slope"] = out["latent_change"].ewm(
            span=params.slope_span, adjust=False, min_periods=params.slope_span
        ).mean()
        out["slope_change"] = out["latent_slope"].diff(params.survival_bars)
        out["relative_atr"] = out["atr"] / close
        out["innovation"] = out["slope_change"] / out["relative_atr"].where(out["relative_atr"] > 0)
        out["innovation_magnitude"] = out["innovation"].abs()
        out["innovation_threshold"] = out["innovation_magnitude"].rolling(
            params.percentile_window, min_periods=params.percentile_window
        ).quantile(params.tail_percentile).shift(1)
        lag = params.survival_bars - 1
        out["candidate_gap"] = out["gap"].shift(lag)
        out["candidate_threshold"] = out["gap_threshold"].shift(lag)
        out["gap_anchor"] = out["previous_close"].shift(lag)
        out["survival_low"] = close.rolling(params.survival_bars).min()
        out["survival_high"] = close.rolling(params.survival_bars).max()
        large = out["candidate_gap"].abs() > out["candidate_threshold"]
        strong = out["innovation_magnitude"] > out["innovation_threshold"]
        bullish = large & strong & (out["candidate_gap"] > 0) & (out["survival_low"] > out["gap_anchor"]) & (out["innovation"] > 0) & (out["latent_slope"] > 0)
        bearish = large & strong & (out["candidate_gap"] < 0) & (out["survival_high"] < out["gap_anchor"]) & (out["innovation"] < 0) & (out["latent_slope"] < 0)
        out["entry"] = np.select([bullish, bearish], [1, -1], default=0).astype(int)
        return out.replace([np.inf, -np.inf], np.nan)

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame,
                         ctx: StrategyContext, params: GapStepParams) -> SignalFrame:
        positions = np.zeros(len(data), dtype=int)
        side = 0
        age = 0
        high_water = low_water = stop = 0.0
        closes = data["close"].to_numpy(dtype=float)
        atrs = indicators["atr"].to_numpy(dtype=float)
        entries = indicators["entry"].fillna(0).to_numpy(dtype=int)
        for i in range(len(data)):
            price, atr = closes[i], atrs[i]
            if not np.isfinite(price) or not np.isfinite(atr) or price <= 0 or atr <= 0:
                positions[i] = side
                if side:
                    age += 1
                continue
            if side:
                age += 1
                high_water = max(high_water, price)
                low_water = min(low_water, price)
                if side == 1:
                    stop = max(stop, high_water - params.trail_atr * atr)
                    stopped = price <= stop
                else:
                    stop = min(stop, low_water + params.trail_atr * atr)
                    stopped = price >= stop
                if stopped or age >= params.holding_bars:
                    side = 0
                    positions[i] = 0
                    continue
            elif entries[i]:
                side = int(entries[i])
                age = 0
                high_water = low_water = price
                stop = price - side * params.trail_atr * atr
            positions[i] = side
        df = pd.DataFrame(index=data.index)
        df["signal"] = positions
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = 1.0
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check():
    strategy = GeneratedStrategy()
    params = GapStepParams(trail_atr=2.5)
    index = pd.date_range("2020-01-01", periods=5)
    for side, prices in [(1, [100, 110, 108, 100, 100]),
                         (-1, [100, 90, 92, 100, 100])]:
        data = pd.DataFrame({"close": prices}, index=index)
        indicators = pd.DataFrame({"atr": 2.0, "entry": [side, 0, 0, 0, 0]}, index=index)
        result = strategy.generate_signals(data, indicators, None, params).data
        assert result["signal"].tolist() == [0, side, side, side, 0]
        assert (result["size"] > 0).all()


if __name__ == "__main__":
    _self_check()
