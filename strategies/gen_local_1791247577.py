from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class AnalogParams:
    fast: int = 5
    slow: int = 20
    volatility: int = 20
    shape_lag: int = 4
    horizon: int = 8
    history: int = 252
    bandwidth: float = 0.65
    threshold: float = 0.10
    smoothing: int = 3

    def __post_init__(self):
        if not (2 <= self.fast < self.slow and self.volatility >= 2):
            raise ValueError("Require 2 <= fast < slow and volatility >= 2")
        if not (self.shape_lag >= 1 and self.horizon >= 1 and self.smoothing >= 1):
            raise ValueError("Lookbacks must be positive")
        if self.history <= self.horizon + 2 * self.shape_lag:
            raise ValueError("History must exceed the analog exclusion window")
        if not (np.isfinite(self.bandwidth) and self.bandwidth > 0):
            raise ValueError("Bandwidth must be finite and positive")
        if not (np.isfinite(self.threshold) and self.threshold >= 0):
            raise ValueError("Threshold must be finite and nonnegative")


class GeneratedStrategy(BaseStrategy[AnalogParams]):
    strategy_id = "gen_local_1791247577"

    @classmethod
    def params_type(cls):
        return AnalogParams

    def warmup_bars(self, params):
        feature_span = max(params.slow, params.volatility) + 2 * params.shape_lag
        outcome_span = params.volatility + params.horizon
        return int(params.history + max(feature_span, outcome_span) + params.smoothing + 1)

    def indicators(self, data, params):
        p = params
        out = pd.DataFrame(index=data.index)
        out["log_close"] = np.log(data["close"].where(data["close"] > 0))
        out["return"] = out["log_close"].diff()
        out["volatility"] = out["return"].rolling(p.volatility).std().replace(0.0, np.nan)
        out["fast_slope"] = out["return"].rolling(p.fast).mean()
        out["slow_slope"] = out["return"].rolling(p.slow).mean()
        out["innovation"] = (out["fast_slope"] - out["slow_slope"]) / out["volatility"]
        out["innovation_middle"] = out["innovation"].shift(p.shape_lag)
        out["innovation_old"] = out["innovation"].shift(2 * p.shape_lag)
        out["trend"] = out["slow_slope"] / out["volatility"]
        out["mature_outcome"] = out["log_close"].diff(p.horizon) / (
            out["volatility"].shift(p.horizon) * np.sqrt(p.horizon)
        )
        features = out[["innovation", "innovation_middle", "innovation_old", "trend"]]
        numerator = pd.Series(0.0, index=data.index)
        denominator = pd.Series(0.0, index=data.index)
        # ponytail: O(bars * history) kernel scan; batch array distances if history grows.
        for lag in range(p.horizon + 2 * p.shape_lag, p.history + 1):
            difference = features - features.shift(lag)
            distance = difference.pow(2).sum(axis=1, min_count=4) / 4.0
            outcome = out["mature_outcome"].shift(lag - p.horizon)
            weight = np.exp(-distance / (2.0 * p.bandwidth ** 2))
            valid = np.isfinite(distance) & np.isfinite(outcome)
            weight = weight.where(valid, 0.0)
            numerator += weight * outcome.where(valid, 0.0)
            denominator += weight
        out["analog_mass"] = denominator
        out["forecast"] = numerator / denominator.where(denominator >= 2.0)
        out["score"] = out["forecast"].rolling(p.smoothing).mean()
        out.loc[np.arange(len(out)) < self.warmup_bars(p) - 1, "score"] = np.nan
        return out.replace([np.inf, -np.inf], np.nan)

    def generate_signals(self, data, indicators, ctx: StrategyContext, params):
        score = indicators["score"].reindex(data.index)
        df = pd.DataFrame(index=data.index)
        # The same condition governs entry and reversal exit; no timed exit.
        df["signal"] = (score > params.threshold).fillna(False).astype(int)
        conviction = (score - params.threshold).clip(lower=0.0).fillna(0.0)
        df["size"] = (0.15 + 0.80 * np.tanh(conviction)).astype(float)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = df["size"].shift(1).fillna(0.15).astype(float)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check():
    p = AnalogParams(history=80)
    index = pd.date_range("2020-01-01", periods=320)
    t = np.arange(len(index))
    close = 100.0 * np.exp(np.cumsum(0.001 + 0.006 * np.sin(t / 9.0)))
    data = pd.DataFrame({
        "open": close, "high": close * 1.01, "low": close * 0.99,
        "close": close, "volume": 1000000.0,
    }, index=index)
    strategy = GeneratedStrategy()
    full = strategy.indicators(data, p)
    prefix = strategy.indicators(data.iloc[:250], p)
    pd.testing.assert_frame_equal(full.iloc[:250], prefix)
    result = strategy.generate_signals(data, full, None, p).data
    expected = (full["score"] > p.threshold).astype(int).shift(1).fillna(0).astype(int)
    pd.testing.assert_series_equal(result["signal"], expected, check_names=False)
    assert result["signal"].isin([0, 1]).all()
    assert np.isfinite(result["size"]).all() and result["size"].between(0.15, 0.95).all()
    assert result["signal"].max() == 1
    assert ((result["signal"].shift(1) == 1) & (result["signal"] == 0)).any()
    flat = data.copy()
    flat[["open", "high", "low", "close"]] = 100.0
    flat_result = strategy.generate_signals(flat, strategy.indicators(flat, p), None, p).data
    assert flat_result["signal"].eq(0).all()


if __name__ == "__main__":
    _self_check()
