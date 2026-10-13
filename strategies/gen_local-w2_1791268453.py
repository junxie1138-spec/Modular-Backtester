from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class GapDrawdownParams:
    drawdown_lookback: int = 20
    atr_period: int = 14
    min_drawdown_atr: float = 2.0
    min_gap_atr: float = 0.25
    trail_atr: float = 1.25
    hold_bars: int = 2


class GeneratedStrategy(BaseStrategy[GapDrawdownParams]):
    strategy_id = "gen_local-w2_1791268453"

    @classmethod
    def params_type(cls):
        return GapDrawdownParams

    def warmup_bars(self, params):
        if params.drawdown_lookback < 1 or params.atr_period < 1:
            raise ValueError("Lookbacks must be positive.")
        if params.hold_bars not in (1, 2):
            raise ValueError("hold_bars must be 1 or 2.")
        thresholds = (params.min_drawdown_atr, params.min_gap_atr, params.trail_atr)
        if any(not np.isfinite(x) or x <= 0 for x in thresholds):
            raise ValueError("ATR thresholds must be finite and positive.")
        return int(max(params.drawdown_lookback, params.atr_period) + 1)

    def indicators(self, data, params):
        self.warmup_bars(params)
        out = pd.DataFrame(index=data.index)
        out["previous_close"] = data["close"].shift(1)
        out["prior_high"] = data["close"].rolling(
            params.drawdown_lookback, min_periods=params.drawdown_lookback
        ).max().shift(1)
        out["true_range"] = pd.concat([
            data["high"] - data["low"],
            (data["high"] - out["previous_close"]).abs(),
            (data["low"] - out["previous_close"]).abs(),
        ], axis=1).max(axis=1)
        out["atr"] = out["true_range"].rolling(
            params.atr_period, min_periods=params.atr_period
        ).mean()
        out["prior_atr"] = out["atr"].shift(1).where(out["atr"].shift(1) > 0)
        out["drawdown_atr"] = (
            out["prior_high"] - out["previous_close"]
        ) / out["prior_atr"]
        out["gap_atr"] = (data["open"] - out["previous_close"]) / out["prior_atr"]
        out["drawdown_ok"] = out["drawdown_atr"].ge(params.min_drawdown_atr)
        out["gap_ok"] = (
            out["gap_atr"].le(-params.min_gap_atr)
            & data["close"].ge(out["previous_close"])
        )
        out["entry"] = (
            out["drawdown_ok"] & out["gap_ok"]
            & out["atr"].gt(0) & np.isfinite(out["atr"])
            & np.isfinite(out["drawdown_atr"]) & np.isfinite(out["gap_atr"])
        ).fillna(False)
        return out

    def generate_signals(self, data, indicators, ctx, params):
        self.warmup_bars(params)
        df = pd.DataFrame(index=data.index)
        signals = np.zeros(len(data), dtype=int)
        stops = np.full(len(data), np.nan)
        highs = np.full(len(data), np.nan)
        active = False
        age = 0
        high_water = -np.inf
        stop = -np.inf
        close_values = data["close"].to_numpy(dtype=float)
        atr_values = indicators["atr"].to_numpy(dtype=float)
        entries = indicators["entry"].fillna(False).to_numpy(dtype=bool)
        for i, (close, atr, entry) in enumerate(zip(close_values, atr_values, entries)):
            valid = np.isfinite(close) and close > 0 and np.isfinite(atr) and atr > 0
            if active:
                age += 1
                if valid:
                    # Entry fills at this bar's open; only subsequent closes enter the watermark.
                    high_water = max(high_water, close)
                    stop = max(stop, high_water - params.trail_atr * atr)
                    highs[i], stops[i] = high_water, stop
                if not valid or close < stop or age >= params.hold_bars:
                    active = False
            elif entry and valid:
                active = True
                age = 0
                high_water = -np.inf
                stop = -np.inf
            signals[i] = int(active)
        df["signal"] = signals
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = 0.95
        df["high_water"] = highs
        df["trailing_stop"] = stops
        return SignalFrame(data=df, signal_column="signal", size_column="size")


if __name__ == "__main__":
    strategy = GeneratedStrategy()
    params = GapDrawdownParams()
    index = pd.date_range("2020-01-01", periods=6)
    close = np.array([100., 101., 99., 100., 100., 100.])
    data = pd.DataFrame({
        "open": close, "high": close + 1, "low": close - 1,
        "close": close, "volume": 1000.,
    }, index=index)
    indicators = pd.DataFrame({
        "atr": [1., 1., 10., 1., 1., 1.],
        "entry": [True, False, False, True, False, False],
    }, index=index)
    ctx = StrategyContext("SPY", "1d", strategy.warmup_bars(params))
    result = strategy.generate_signals(data, indicators, ctx, params).data
    assert result["signal"].tolist() == [0, 1, 1, 0, 1, 1]
    assert result["trailing_stop"].iloc[1] == result["trailing_stop"].iloc[2] == 99.75
    assert result["high_water"].iloc[2] == 101.
    params.hold_bars = 1
    result = strategy.generate_signals(data, indicators, ctx, params).data
    assert result["signal"].tolist() == [0, 1, 0, 0, 1, 0]
    assert not strategy.indicators(data, params)["entry"].any()
