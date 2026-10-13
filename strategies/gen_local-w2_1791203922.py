from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class VisibilityParams:
    graph_window: int = 20
    range_window: int = 40
    atr_window: int = 14
    hub_threshold: float = 0.30
    extreme_fraction: float = 0.20
    trail_atr: float = 3.0
    hold_bars: int = 18


class GeneratedStrategy(BaseStrategy[VisibilityParams]):
    strategy_id = "gen_local-w2_1791203922"

    @classmethod
    def params_type(cls) -> type[VisibilityParams]:
        return VisibilityParams

    @staticmethod
    def warmup_bars(params: VisibilityParams) -> int:
        return int(max(params.graph_window, params.range_window, params.atr_window + 1))

    @staticmethod
    def indicators(data: pd.DataFrame, params: VisibilityParams) -> pd.DataFrame:
        if min(params.graph_window, params.range_window) < 3 or params.atr_window < 1:
            raise ValueError("Invalid indicator windows")
        if not (0 < params.hub_threshold < 1 and 0 < params.extreme_fraction < 0.5):
            raise ValueError("Invalid entry thresholds")
        if not np.isfinite(params.trail_atr) or params.trail_atr <= 0 or params.hold_bars < 1:
            raise ValueError("Invalid exit parameters")
        close = data["close"].where(data["close"] > 0).replace([np.inf, -np.inf], np.nan)
        log_close = np.log(close)
        upper_degree = np.zeros(len(data), dtype=float)
        lower_degree = np.zeros(len(data), dtype=float)
        upper_blocker = np.full(len(data), -np.inf)
        lower_blocker = np.full(len(data), -np.inf)
        # Endpoint natural-visibility edges are record slopes to earlier nodes.
        for lag in range(1, params.graph_window):
            slope = ((log_close.shift(lag) - log_close) / lag).to_numpy()
            finite = np.isfinite(slope)
            upper_degree += finite & (slope > upper_blocker)
            lower_degree += finite & (-slope > lower_blocker)
            upper_blocker = np.maximum(upper_blocker, np.where(finite, slope, -np.inf))
            lower_blocker = np.maximum(lower_blocker, np.where(finite, -slope, -np.inf))
        ready = close.rolling(params.graph_window).count().eq(params.graph_window)
        upper = pd.Series(upper_degree, index=data.index).where(ready)
        lower = pd.Series(lower_degree, index=data.index).where(ready)
        range_low = close.rolling(params.range_window).min()
        range_high = close.rolling(params.range_window).max()
        span = (range_high - range_low).replace(0, np.nan)
        previous_close = close.shift(1)
        true_range = pd.concat(
            [data["high"] - data["low"],
             (data["high"] - previous_close).abs(),
             (data["low"] - previous_close).abs()], axis=1
        ).max(axis=1).where(previous_close.notna())
        true_range = true_range.replace([np.inf, -np.inf], np.nan).where(true_range >= 0)
        return pd.DataFrame({
            "log_close": log_close,
            "upper_degree": upper,
            "lower_degree": lower,
            "hub_balance": (upper - lower) / (upper + lower).replace(0, np.nan),
            "range_low": range_low,
            "range_high": range_high,
            "relative_position": (close - range_low) / span,
            "true_range": true_range,
            "atr": true_range.rolling(params.atr_window).mean()
        }, index=data.index)

    @staticmethod
    def generate_signals(
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: VisibilityParams
    ) -> SignalFrame:
        close = data["close"].to_numpy(dtype=float)
        atr = indicators["atr"].to_numpy(dtype=float)
        hub = indicators["hub_balance"].to_numpy(dtype=float)
        location = indicators["relative_position"].to_numpy(dtype=float)
        desired = np.zeros(len(data), dtype=int)
        position = 0
        age = 0
        high_water = low_water = np.nan
        high_stop = -np.inf
        short_stop = np.inf
        for i in range(len(data)):
            valid_price = np.isfinite(close[i]) and close[i] > 0
            valid_atr = np.isfinite(atr[i]) and atr[i] > 0
            if position:
                age += 1
                if not (valid_price and valid_atr):
                    position = 0
                else:
                    # First held close is the bar after the entry decision.
                    high_water = close[i] if age == 1 else max(high_water, close[i])
                    low_water = close[i] if age == 1 else min(low_water, close[i])
                    high_stop = max(high_stop, high_water - params.trail_atr * atr[i])
                    # Shorts also have a mirrored protective stop against rallies.
                    short_stop = min(short_stop, low_water + params.trail_atr * atr[i])
                    # The required high-water close-drop exit applies to both sides.
                    if (close[i] <= high_stop or
                            (position == -1 and close[i] >= short_stop) or
                            age >= params.hold_bars):
                        position = 0
                # An exit always emits a flat bar; no same-bar re-entry.
            elif valid_price and valid_atr and np.isfinite(hub[i]) and np.isfinite(location[i]):
                if hub[i] >= params.hub_threshold and location[i] <= params.extreme_fraction:
                    position = 1
                elif hub[i] <= -params.hub_threshold and location[i] >= 1 - params.extreme_fraction:
                    position = -1
                if position:
                    age = 0
                    high_water = low_water = np.nan
                    high_stop = -np.inf
                    short_stop = np.inf
            desired[i] = position
        df = pd.DataFrame({"signal": desired, "size": 0.95}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check() -> None:
    params = VisibilityParams()
    dates = pd.date_range("2020-01-01", periods=25)
    data = pd.DataFrame({"close": 100.0}, index=dates)
    ind = pd.DataFrame({"atr": 2.0, "hub_balance": 0.0, "relative_position": 0.5}, index=dates)
    ind.iloc[0] = [2.0, 1.0, 0.0]
    result = GeneratedStrategy.generate_signals(data, ind, None, params).data
    assert result["signal"].tolist() == [0] + [1] * 18 + [0] * 6
    data.iloc[:5, 0] = [100, 100, 110, 108, 103]
    ind.loc[dates[3]:, "atr"] = 20.0
    result = GeneratedStrategy.generate_signals(data, ind, None, params).data
    assert result["signal"].iloc[:7].tolist() == [0, 1, 1, 1, 1, 0, 0]
    data["close"] = 100.0
    data.iloc[2, 0] = 90.0
    ind["atr"] = 2.0
    ind.iloc[0] = [2.0, -1.0, 1.0]
    result = GeneratedStrategy.generate_signals(data, ind, None, params).data
    assert result["signal"].iloc[:5].tolist() == [0, -1, -1, 0, 0]


if __name__ == "__main__":
    _self_check()
