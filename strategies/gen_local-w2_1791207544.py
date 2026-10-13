from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class DrawdownChangeParams:
    peak_lookback: int = 40
    atr_period: int = 14
    min_depth_atr: float = 2.0
    deepening_step_atr: float = 0.25
    change_threshold: float = 3.0
    trail_atr: float = 1.2
    hold_bars: int = 2


class GeneratedStrategy(BaseStrategy[DrawdownChangeParams]):
    strategy_id = "gen_local-w2_1791207544"

    @classmethod
    def params_type(cls):
        return DrawdownChangeParams

    def warmup_bars(self, params: DrawdownChangeParams) -> int:
        return int(max(params.peak_lookback, params.atr_period) + 1)

    def indicators(self, data: pd.DataFrame, params: DrawdownChangeParams) -> pd.DataFrame:
        thresholds = (params.min_depth_atr, params.deepening_step_atr,
                      params.change_threshold, params.trail_atr)
        if (params.peak_lookback < 2 or params.atr_period < 2
                or params.hold_bars not in (1, 2)
                or not all(np.isfinite(x) and x > 0 for x in thresholds)):
            raise ValueError("Positive thresholds, lookbacks >= 2, and hold_bars in (1, 2) required.")
        out = pd.DataFrame(index=data.index)
        close = data["close"].where(data["close"] > 0)
        previous = close.shift(1)
        true_range = pd.concat([
            data["high"] - data["low"],
            (data["high"] - previous).abs(),
            (data["low"] - previous).abs(),
        ], axis=1).max(axis=1)
        true_range = true_range.where(previous.notna())
        out["true_range"] = true_range
        out["atr"] = true_range.rolling(params.atr_period, min_periods=params.atr_period).mean()
        out["atr"] = out["atr"].where(out["atr"] > 0)
        out["peak"] = close.rolling(params.peak_lookback, min_periods=params.peak_lookback).max()
        out["depth"] = (out["peak"] - close).clip(lower=0)
        out["depth_atr"] = out["depth"] / out["atr"]
        out["depth_change"] = out["depth"].diff() / out["atr"].shift(1)
        # Sequential log likelihood: deepening probability changes from 0.7 to 0.3.
        out["log_evidence"] = pd.Series(
            np.where(out["depth_change"] > params.deepening_step_atr,
                     np.log(0.3 / 0.7), np.log(0.7 / 0.3)),
            index=data.index,
        ).where(out["depth_change"].notna())
        return out.replace([np.inf, -np.inf], np.nan)

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: DrawdownChangeParams,
    ) -> SignalFrame:
        df = pd.DataFrame({"signal": 0, "size": 1.0}, index=data.index)
        signals = np.zeros(len(data), dtype=int)
        sizes = np.ones(len(data), dtype=float)
        closes = data["close"].to_numpy(dtype=float)
        values = indicators[["atr", "peak", "depth_atr", "depth_change",
                             "log_evidence"]].to_numpy(dtype=float)
        armed = False
        evidence = 0.0
        entry_bar = -1
        high_water = -np.inf
        stop = -np.inf
        entry_size = 1.0
        for i in range(len(data)):
            close = closes[i]
            atr, peak, depth, change, increment = values[i]
            valid = np.isfinite(close) and close > 0 and np.isfinite(values[i]).all() and atr > 0
            if entry_bar >= 0:
                if valid:
                    # The first in-trade close occurs one bar after the entry decision.
                    high_water = max(high_water, close)
                    stop = max(stop, high_water - params.trail_atr * atr)
                if not valid or close <= stop or i - entry_bar >= params.hold_bars:
                    entry_bar = -1
                    armed = False
                    evidence = 0.0
                    high_water = stop = -np.inf
                else:
                    signals[i] = 1
                    sizes[i] = entry_size
                continue
            stable_anchor = i > 0 and np.isfinite(values[i - 1, 1]) and peak == values[i - 1, 1]
            if not valid or not stable_anchor or depth < params.min_depth_atr:
                armed = False
                evidence = 0.0
                continue
            if not armed:
                if change > params.deepening_step_atr:
                    armed = True
                    evidence = 0.0
                continue
            evidence = max(0.0, evidence + increment)
            if evidence >= params.change_threshold:
                strength = (1.0 - np.exp(-evidence / params.change_threshold))
                strength *= min(1.0, depth / (2.0 * params.min_depth_atr))
                entry_size = float(np.clip(0.25 + 0.75 * strength, 0.25, 1.0))
                entry_bar = i
                high_water = stop = -np.inf
                signals[i] = 1
                sizes[i] = entry_size
                armed = False
                evidence = 0.0
        df["signal"] = signals
        df["size"] = sizes
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = df["size"].shift(1).fillna(1.0).astype(float)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


if __name__ == "__main__":
    strategy = GeneratedStrategy()
    params = DrawdownChangeParams(change_threshold=1.0)
    index = pd.date_range("2020-01-01", periods=8)
    data = pd.DataFrame({
        "open": 98.0, "high": 100.0, "low": 96.0,
        "close": [98.0, 98.0, 98.0, 98.0, 99.0, 97.0, 98.0, 98.0],
        "volume": 1000.0,
    }, index=index)
    change = pd.Series([0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], index=index)
    indicators = pd.DataFrame({
        "atr": 1.0, "peak": 100.0, "depth_atr": 3.0, "depth_change": change,
        "log_evidence": np.where(change > params.deepening_step_atr, np.log(0.3 / 0.7), np.log(0.7 / 0.3)),
    }, index=index)
    ctx = StrategyContext(symbol="SPY", timeframe="1d", warmup_bars=0)
    result = strategy.generate_signals(data, indicators, ctx, params).data
    assert result["signal"].tolist() == [0, 0, 0, 0, 1, 1, 0, 0]
    assert result["size"].between(0.25, 1.0).all()
    assert result["size"].iloc[4] == result["size"].iloc[5] < 1.0
    prefix = strategy.generate_signals(data.iloc[:6], indicators.iloc[:6], ctx, params).data
    pd.testing.assert_frame_equal(prefix, result.iloc[:6])
    params.hold_bars = 1
    shorter = strategy.generate_signals(data, indicators, ctx, params).data
    assert shorter["signal"].tolist() == [0, 0, 0, 0, 1, 0, 0, 0]
