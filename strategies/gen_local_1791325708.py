from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class Params:
    frac_order: float = 0.45
    frac_window: int = 32
    breakout_window: int = 20
    atr_window: int = 14
    trail_atr: float = 1.8
    hold_bars: int = 4


class GeneratedStrategy(BaseStrategy[Params]):
    strategy_id = "gen_local_1791325708"

    @classmethod
    def params_type(cls) -> type[Params]:
        return Params

    def warmup_bars(self, params: Params) -> int:
        return int(max(params.frac_window + params.breakout_window + 1, params.atr_window + 1))

    def indicators(self, data: pd.DataFrame, params: Params) -> pd.DataFrame:
        if not 0.0 < params.frac_order < 1.0:
            raise ValueError("frac_order must be between zero and one")
        if min(params.frac_window, params.breakout_window, params.atr_window) < 2:
            raise ValueError("indicator windows must be at least two")
        if not np.isfinite(params.trail_atr) or params.trail_atr <= 0 or not 3 <= params.hold_bars <= 5:
            raise ValueError("trail_atr must be positive and hold_bars must be 3 to 5")
        close = data["close"].astype(float).replace([np.inf, -np.inf], np.nan)
        out = pd.DataFrame(index=data.index)
        weights = np.ones(params.frac_window, dtype=float)
        for lag in range(1, params.frac_window):
            weights[lag] = -weights[lag - 1] * (params.frac_order - lag + 1) / lag
        # ponytail: fixed finite memory; increase frac_window if truncation sensitivity matters.
        fractional = close.copy()
        for lag in range(1, params.frac_window):
            fractional = fractional + weights[lag] * close.shift(lag)
        out["fractional_close"] = fractional
        out["upper"] = fractional.rolling(params.breakout_window, min_periods=params.breakout_window).max().shift(1)
        out["lower"] = fractional.rolling(params.breakout_window, min_periods=params.breakout_window).min().shift(1)
        previous = close.shift(1)
        ranges = pd.concat([
            data["high"] - data["low"],
            (data["high"] - previous).abs(),
            (data["low"] - previous).abs(),
        ], axis=1)
        out["true_range"] = ranges.max(axis=1, skipna=False).replace([np.inf, -np.inf], np.nan)
        out["atr"] = out["true_range"].rolling(params.atr_window, min_periods=params.atr_window).mean()
        out["up_breakout"] = out["fractional_close"].gt(out["upper"])
        out["down_breakout"] = out["fractional_close"].lt(out["lower"])
        out["long_confirmed"] = out["up_breakout"] & out["up_breakout"].shift(1, fill_value=False)
        out["short_confirmed"] = out["down_breakout"] & out["down_breakout"].shift(1, fill_value=False)
        out["long_entry"] = out["long_confirmed"] & ~out["long_confirmed"].shift(1, fill_value=False)
        out["short_entry"] = out["short_confirmed"] & ~out["short_confirmed"].shift(1, fill_value=False)
        return out

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: Params) -> SignalFrame:
        close = data["close"].to_numpy(dtype=float)
        atr = indicators["atr"].to_numpy(dtype=float)
        longs = indicators["long_entry"].fillna(False).to_numpy(dtype=bool)
        shorts = indicators["short_entry"].fillna(False).to_numpy(dtype=bool)
        raw = np.zeros(len(data), dtype=int)
        position, age = 0, 0
        extreme = stop = np.nan
        for i, (price, volatility) in enumerate(zip(close, atr)):
            if not np.isfinite(price) or not np.isfinite(volatility) or volatility <= 0:
                position, age = 0, 0
                extreme = stop = np.nan
            elif position:
                # The previous decision fills at this bar's open; this close is in-trade.
                age += 1
                if position == 1:
                    extreme = price if np.isnan(extreme) else max(extreme, price)
                    candidate = extreme - params.trail_atr * volatility
                    stop = candidate if np.isnan(stop) else max(stop, candidate)
                    stopped = price <= stop
                else:
                    # Mirror the high-water trailing exit for short positions.
                    extreme = price if np.isnan(extreme) else min(extreme, price)
                    candidate = extreme + params.trail_atr * volatility
                    stop = candidate if np.isnan(stop) else min(stop, candidate)
                    stopped = price >= stop
                if stopped or age >= params.hold_bars:
                    position, age = 0, 0
                    extreme = stop = np.nan
            elif longs[i] or shorts[i]:
                position = 1 if longs[i] else -1
                age = 0
                extreme = stop = np.nan
            raw[i] = position
        df = pd.DataFrame({"signal": raw, "size": 0.95}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check() -> None:
    strategy = GeneratedStrategy()
    params = Params()
    index = pd.date_range("2020-01-01", periods=6, freq="D")
    for direction, prices in [(1, [100, 100, 103, 102, 100, 100]), (-1, [100, 100, 97, 98, 100, 100])]:
        data = pd.DataFrame({"close": prices}, index=index)
        indicators = pd.DataFrame({
            "atr": [1, 1, 1, 10, 1, 1],
            "long_entry": [direction == 1, False, False, False, False, False],
            "short_entry": [direction == -1, False, False, False, False, False],
        }, index=index)
        result = strategy.generate_signals(data, indicators, None, params).data
        assert result["signal"].tolist() == [0, direction, direction, direction, direction, 0]
        params.hold_bars = 5
        result = strategy.generate_signals(data, indicators, None, params).data
        assert result["signal"].tolist() == [0, direction, direction, direction, direction, 0]
        params.hold_bars = 4
    data = pd.DataFrame({"close": [100.0] * 6}, index=index)
    indicators["atr"] = 1.0
    indicators["long_entry"] = [True, False, False, False, False, False]
    indicators["short_entry"] = False
    assert strategy.generate_signals(data, indicators, None, params).data["signal"].tolist() == [0, 1, 1, 1, 1, 0]
    index = pd.date_range("2020-01-01", periods=100, freq="D")
    prices = 100.0 + np.arange(100, dtype=float) ** 2 / 100.0
    data = pd.DataFrame({"open": prices, "high": prices + 1, "low": prices - 1, "close": prices, "volume": 1000.0}, index=index)
    derived = strategy.indicators(data, params)
    assert derived["long_entry"].any()
    assert (derived["long_entry"] <= (derived["up_breakout"] & derived["up_breakout"].shift(1, fill_value=False))).all()
    pd.testing.assert_frame_equal(derived.iloc[:80], strategy.indicators(data.iloc[:80], params))
    assert strategy.warmup_bars(params) >= params.frac_window + params.breakout_window + 1


if __name__ == "__main__":
    _self_check()
