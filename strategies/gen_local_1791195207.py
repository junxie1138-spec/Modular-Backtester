from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class FractionalFloorParams:
    fractional_order: float = 0.45
    memory_bars: int = 60
    floor_bars: int = 15
    percentile_bars: int = 126
    entry_percentile: float = 0.75
    atr_bars: int = 14
    trail_atr: float = 2.5
    max_hold_bars: int = 20


class GeneratedStrategy(BaseStrategy[FractionalFloorParams]):
    strategy_id = "gen_local_1791195207"

    @classmethod
    def params_type(cls):
        return FractionalFloorParams

    def warmup_bars(self, params: FractionalFloorParams) -> int:
        return int(max(
            params.memory_bars + params.percentile_bars,
            params.memory_bars + params.floor_bars - 1,
            params.atr_bars + 1,
        ))

    def indicators(self, data: pd.DataFrame, params: FractionalFloorParams) -> pd.DataFrame:
        if not (
            0.0 < params.fractional_order < 1.0
            and params.memory_bars >= 2
            and params.floor_bars >= 1
            and params.percentile_bars >= 2
            and 0.0 < params.entry_percentile < 1.0
            and params.atr_bars >= 1
            and params.trail_atr > 0.0
            and params.max_hold_bars >= 1
        ):
            raise ValueError("Invalid strategy parameters")
        close = data["close"].astype(float).replace([np.inf, -np.inf], np.nan)
        close = close.where(close > 0.0)
        fractional_close = close.copy()
        weight = 1.0
        # ponytail: finite memory truncates the fractional kernel; increase memory_bars for longer tails.
        for lag in range(1, params.memory_bars):
            weight *= (lag - 1.0 - params.fractional_order) / lag
            fractional_close = fractional_close + weight * close.shift(lag)
        fractional_strength = (fractional_close / close).replace([np.inf, -np.inf], np.nan)
        floor = fractional_strength.rolling(
            params.floor_bars, min_periods=params.floor_bars
        ).min()
        threshold = fractional_strength.shift(1).rolling(
            params.percentile_bars, min_periods=params.percentile_bars
        ).quantile(params.entry_percentile)
        previous_close = close.shift(1)
        true_range = pd.concat([
            data["high"] - data["low"],
            (data["high"] - previous_close).abs(),
            (data["low"] - previous_close).abs(),
        ], axis=1).max(axis=1, skipna=False)
        true_range = true_range.where(np.isfinite(true_range) & (true_range >= 0.0))
        atr = true_range.rolling(params.atr_bars, min_periods=params.atr_bars).mean()
        return pd.DataFrame({
            "fractional_close": fractional_close,
            "fractional_strength": fractional_strength,
            "strength_floor": floor,
            "entry_threshold": threshold,
            "true_range": true_range,
            "atr": atr,
        }, index=data.index)

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: FractionalFloorParams,
    ) -> SignalFrame:
        close = data["close"].to_numpy(dtype=float)
        atr = indicators["atr"].to_numpy(dtype=float)
        entry = (
            indicators["strength_floor"] > indicators["entry_threshold"]
        ).fillna(False).to_numpy(dtype=bool)
        signals = np.zeros(len(data), dtype=int)
        holding = False
        high_water = -np.inf
        stop = -np.inf
        age = 0
        for i in range(len(data)):
            valid = np.isfinite(close[i]) and close[i] > 0.0 and np.isfinite(atr[i]) and atr[i] > 0.0
            if holding:
                # The first held bar follows the entry decision, matching the final shift.
                age += 1
                if valid:
                    high_water = max(high_water, close[i])
                    stop = max(stop, high_water - params.trail_atr * atr[i])
                if not valid or close[i] <= stop or age >= params.max_hold_bars:
                    holding = False
                    high_water = -np.inf
                    stop = -np.inf
                    age = 0
            elif valid and entry[i]:
                holding = True
                high_water = -np.inf
                stop = -np.inf
                age = 0
            signals[i] = int(holding)
        df = pd.DataFrame({"signal": signals, "size": 0.95}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


if __name__ == "__main__":
    strategy = GeneratedStrategy()
    params = FractionalFloorParams(trail_atr=2.0)
    index = pd.date_range("2020-01-01", periods=6)
    close = pd.Series([100.0, 101.0, 103.0, 104.0, 101.0, 105.0], index=index)
    data = pd.DataFrame({
        "open": close, "high": close + 1.0, "low": close - 1.0,
        "close": close, "volume": 1000.0,
    }, index=index)
    indicators = pd.DataFrame({
        "strength_floor": 1.0, "entry_threshold": 0.0,
        "atr": [1.0, 1.0, 1.0, 10.0, 10.0, 1.0],
    }, index=index)
    ctx = StrategyContext("SPY", "1d", strategy.warmup_bars(params))
    result = strategy.generate_signals(data, indicators, ctx, params).data
    assert result["signal"].tolist() == [0, 1, 1, 1, 1, 0]
    params.max_hold_bars = 3
    indicators["atr"] = 1.0
    result = strategy.generate_signals(data, indicators, ctx, params).data
    assert result["signal"].tolist() == [0, 1, 1, 1, 0, 1]
    assert strategy.indicators(data, params)["entry_threshold"].isna().all()
