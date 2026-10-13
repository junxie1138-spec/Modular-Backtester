from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class OrdinalCompressionParams:
    pattern_window: int = 24
    compression_window: int = 10
    baseline_window: int = 60
    atr_window: int = 14
    pressure_window: int = 5
    complexity_ceiling: float = 0.55
    motif_threshold: float = 0.15
    compression_ceiling: float = 0.80
    pressure_threshold: float = 0.20
    stop_atr: float = 3.0
    hold_bars: int = 20


class GeneratedStrategy(BaseStrategy[OrdinalCompressionParams]):
    strategy_id = "gen_local_1791239858"

    @classmethod
    def params_type(cls):
        return OrdinalCompressionParams

    def warmup_bars(self, params):
        return int(max(params.pattern_window + 3, params.baseline_window + 1,
                       params.compression_window + 1, params.atr_window + 1,
                       params.pressure_window + 1))

    def indicators(self, data, params):
        windows = (params.pattern_window, params.compression_window,
                   params.baseline_window, params.atr_window,
                   params.pressure_window, params.hold_bars)
        if any(w < 1 for w in windows) or params.stop_atr <= 0:
            raise ValueError("Windows and stop_atr must be positive.")
        out = pd.DataFrame(index=data.index)
        close = data["close"]
        previous = close.shift(1)
        out["true_range"] = pd.concat(
            [data["high"] - data["low"],
             (data["high"] - previous).abs(),
             (data["low"] - previous).abs()], axis=1
        ).max(axis=1).where(previous.notna())
        out["atr"] = out["true_range"].rolling(params.atr_window).mean()
        out["fast_range"] = out["true_range"].rolling(
            params.compression_window).mean().shift(1)
        out["slow_range"] = out["true_range"].rolling(
            params.baseline_window).mean().shift(1)
        out["compression"] = out["fast_range"] / out["slow_range"].where(
            out["slow_range"] > 0)
        a, b, c = close.shift(2), previous, close
        # Chronological tie breaking yields six deterministic ordinal states.
        out["pattern"] = (
            4 * (a <= b).astype(int) + 2 * (b <= c).astype(int)
            + (a <= c).astype(int)
        ).astype(float).where(a.notna() & b.notna() & c.notna())
        out["transition"] = out["pattern"].shift(1) * 8 + out["pattern"]
        valid = out["transition"].notna()
        states = (0, 2, 3, 4, 5, 7)
        joint_entropy = pd.Series(0.0, index=data.index)
        marginal_entropy = pd.Series(0.0, index=data.index)
        for source in states:
            marginal = pd.Series(0.0, index=data.index)
            for target in states:
                column = "p_" + str(source) + "_" + str(target)
                probability = (out["transition"] == source * 8 + target
                               ).astype(float).where(valid).rolling(
                                   params.pattern_window).mean()
                out[column] = probability
                marginal = marginal + probability
                joint_entropy = joint_entropy - probability * np.log(
                    probability.where(probability > 0, 1.0))
            out["source_" + str(source)] = marginal
            marginal_entropy = marginal_entropy - marginal * np.log(
                marginal.where(marginal > 0, 1.0))
        out["joint_entropy"] = joint_entropy
        out["source_entropy"] = marginal_entropy
        out["complexity"] = ((joint_entropy - marginal_entropy) /
                             np.log(6.0)).clip(0.0, 1.0)
        out["motif_bias"] = (
            (out["pattern"] == 7).astype(float)
            - (out["pattern"] == 0).astype(float)
        ).where(out["pattern"].notna()).rolling(params.pattern_window).mean()
        out["pressure_range"] = out["true_range"].rolling(
            params.pressure_window).sum()
        out["pressure"] = close.diff(params.pressure_window) / out[
            "pressure_range"].where(out["pressure_range"] > 0)
        return out.replace([np.inf, -np.inf], np.nan)

    def generate_signals(self, data, indicators, ctx, params):
        ready = indicators[
            ["atr", "complexity", "motif_bias", "compression", "pressure"]
        ].notna().all(axis=1) & (indicators["atr"] > 0)
        grammar = indicators["complexity"] <= params.complexity_ceiling
        compressed = indicators["compression"] <= params.compression_ceiling
        long_entry = (ready & grammar & compressed
                      & (indicators["motif_bias"] >= params.motif_threshold)
                      & (indicators["pressure"] >= params.pressure_threshold))
        short_entry = (ready & grammar & compressed
                       & (indicators["motif_bias"] <= -params.motif_threshold)
                       & (indicators["pressure"] <= -params.pressure_threshold))
        entries = np.where(long_entry, 1, np.where(short_entry, -1, 0))
        closes = data["close"].to_numpy(dtype=float)
        atrs = indicators["atr"].to_numpy(dtype=float)
        signals = np.zeros(len(data), dtype=int)
        position = 0
        entry_bar = -1
        high_water = low_water = np.nan
        trailing_floor = -np.inf
        trailing_ceiling = np.inf
        for i in range(len(data)):
            if position:
                close, atr = closes[i], atrs[i]
                if not np.isfinite(close) or not np.isfinite(atr) or atr <= 0:
                    position = 0
                else:
                    if i == entry_bar:
                        high_water = low_water = close
                    high_water = max(high_water, close)
                    low_water = min(low_water, close)
                    trailing_floor = max(
                        trailing_floor, high_water - params.stop_atr * atr)
                    trailing_ceiling = min(
                        trailing_ceiling, low_water + params.stop_atr * atr)
                    # Required high-water drawdown exit applies to both sides.
                    # Shorts also exit on a rebound from their in-trade low.
                    stopped = close <= trailing_floor or (
                        position == -1 and close >= trailing_ceiling)
                    expired = i - entry_bar + 1 >= params.hold_bars
                    if stopped or expired:
                        position = 0
            elif entries[i]:
                position = int(entries[i])
                entry_bar = i + 1
                high_water = low_water = np.nan
                trailing_floor = -np.inf
                trailing_ceiling = np.inf
            signals[i] = position
        df = pd.DataFrame({"signal": signals, "size": 0.95}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


if __name__ == "__main__":
    index = pd.date_range("2020-01-01", periods=7)
    data = pd.DataFrame({"open": [100, 100, 105, 104, 102, 102, 102],
                         "high": [101, 101, 106, 105, 103, 103, 103],
                         "low": [99, 99, 104, 103, 101, 101, 101],
                         "close": [100, 100, 105, 104, 102, 102, 102],
                         "volume": 1000.0}, index=index)
    indicators = pd.DataFrame({"atr": [1, 1, 1, 10, 1, 1, 1],
                               "complexity": 0.1,
                               "motif_bias": [0.5, 0, 0, 0, 0, 0, 0],
                               "compression": 0.5,
                               "pressure": [0.5, 0, 0, 0, 0, 0, 0]}, index=index)
    strategy = GeneratedStrategy()
    params = OrdinalCompressionParams(stop_atr=2.0)
    ctx = StrategyContext("SPY", "1d", strategy.warmup_bars(params))
    result = strategy.generate_signals(data, indicators, ctx, params).data
    assert result["signal"].tolist() == [0, 1, 1, 1, 1, 0, 0]
    indicators.loc[index[0], ["motif_bias", "pressure"]] = -0.5
    result = strategy.generate_signals(data, indicators, ctx, params).data
    assert result["signal"].tolist() == [0, -1, -1, 0, 0, 0, 0]
    indicators.loc[index[0], ["motif_bias", "pressure"]] = 0.5
    params.hold_bars = 2
    result = strategy.generate_signals(data, indicators, ctx, params).data
    assert result["signal"].tolist() == [0, 1, 1, 0, 0, 0, 0]
