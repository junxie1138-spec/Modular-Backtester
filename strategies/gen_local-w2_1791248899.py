from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class DrawdownAnalogParams:
    peak_window: int = 20
    shape_length: int = 6
    history: int = 504
    neighbors: int = 12
    horizon: int = 8
    atr_window: int = 14
    threshold_window: int = 126
    entry_percentile: float = 0.80
    depth_percentile: float = 0.65
    atr_multiple: float = 2.5

    def __post_init__(self):
        windows = (self.peak_window, self.shape_length, self.history,
                   self.neighbors, self.horizon, self.atr_window,
                   self.threshold_window)
        if any(not isinstance(x, int) or isinstance(x, bool) or x < 1 for x in windows):
            raise ValueError("Lookbacks, neighbors and horizon must be positive integers.")
        if self.history < self.shape_length + self.horizon * (self.neighbors + 1):
            raise ValueError("History must contain enough separated analog outcomes.")
        if not (0 < self.entry_percentile < 1 and 0 < self.depth_percentile < 1):
            raise ValueError("Percentiles must lie strictly between zero and one.")
        if not np.isfinite(self.atr_multiple) or self.atr_multiple <= 0:
            raise ValueError("ATR multiple must be finite and positive.")


class GeneratedStrategy(BaseStrategy[DrawdownAnalogParams]):
    strategy_id = "gen_local-w2_1791248899"

    @classmethod
    def params_type(cls):
        return DrawdownAnalogParams

    def warmup_bars(self, params):
        return int(max(params.peak_window, params.atr_window + 1)
                   + params.shape_length + params.history + params.horizon
                   + params.threshold_window)

    def indicators(self, data, params):
        p = params
        out = pd.DataFrame(index=data.index)
        close = data["close"].where(data["close"] > 0)
        previous = close.shift(1)
        out["true_range"] = pd.concat(
            [data["high"] - data["low"],
             (data["high"] - previous).abs(),
             (data["low"] - previous).abs()], axis=1
        ).max(axis=1)
        out["atr"] = out["true_range"].rolling(p.atr_window).mean()
        out["peak"] = close.rolling(p.peak_window).max()
        out["depth"] = (1.0 - close / out["peak"]).clip(lower=0)
        out["depth_atr"] = (out["peak"] - close) / out["atr"].where(out["atr"] > 0)
        columns = []
        for lag in range(p.shape_length):
            name = f"depth_lag_{lag}"
            out[name] = out["depth_atr"].shift(lag)
            columns.append(name)
        shapes = out[columns].to_numpy(dtype=float)
        prices = close.to_numpy(dtype=float)
        forecast = np.full(len(data), np.nan)
        score = np.full(len(data), np.nan)
        start = max(p.peak_window, p.atr_window + 1) + p.shape_length + p.history + p.horizon
        # ponytail: bounded brute-force neighbors; batch distances if histories grow materially.
        for t in range(start, len(data)):
            if not np.isfinite(shapes[t]).all():
                continue
            # Outcome windows are separated; the embargo excludes the current shape.
            candidates = np.arange(t - p.history, t - p.horizon - p.shape_length + 1, p.horizon)
            outcomes = prices[candidates + p.horizon] / prices[candidates] - 1.0
            valid = np.isfinite(shapes[candidates]).all(axis=1) & np.isfinite(outcomes)
            candidates, outcomes = candidates[valid], outcomes[valid]
            if len(candidates) < p.neighbors:
                continue
            distance = np.mean((shapes[candidates] - shapes[t]) ** 2, axis=1)
            nearest = np.argsort(distance, kind="stable")[:p.neighbors]
            weights = 1.0 / (1.0 + distance[nearest])
            returns = outcomes[nearest]
            forecast[t] = np.average(returns, weights=weights)
            rms = np.sqrt(np.average(returns ** 2, weights=weights))
            score[t] = forecast[t] / rms if rms > 0 else 0.0
        out["forecast"] = forecast
        out["score"] = score
        out["score_threshold"] = out["score"].abs().shift(1).rolling(
            p.threshold_window
        ).quantile(p.entry_percentile)
        out["depth_threshold"] = out["depth"].shift(1).rolling(
            p.threshold_window
        ).quantile(p.depth_percentile)
        return out.replace([np.inf, -np.inf], np.nan)

    def generate_signals(self, data, indicators, ctx: StrategyContext, params):
        p = params
        ready = np.isfinite(indicators[
            ["atr", "depth", "depth_threshold", "score", "score_threshold"]
        ]).all(axis=1) & (indicators["atr"] > 0)
        eligible = (ready & (indicators["depth"] > indicators["depth_threshold"])
                    & (indicators["score"].abs() > indicators["score_threshold"]))
        entry = np.where(eligible, np.sign(indicators["score"].fillna(0)), 0).astype(int)
        closes = data["close"].to_numpy(dtype=float)
        atr = indicators["atr"].to_numpy(dtype=float)
        signals = np.zeros(len(data), dtype=int)
        position, entered = 0, -1
        high, low, stop, short_stop = np.nan, np.nan, -np.inf, np.inf
        for t in range(len(data)):
            if position:
                if not np.isfinite(closes[t]) or not np.isfinite(atr[t]) or atr[t] <= 0:
                    position = 0
                else:
                    # The first in-trade close is on the bar after the entry decision.
                    high = closes[t] if np.isnan(high) else max(high, closes[t])
                    low = closes[t] if np.isnan(low) else min(low, closes[t])
                    stop = max(stop, high - p.atr_multiple * atr[t])
                    short_stop = min(short_stop, low + p.atr_multiple * atr[t])
                    # Required highest-close exit applies to both sides.
                    # Shorts also have a mirrored adverse-price trailing exit.
                    if (closes[t] <= stop
                            or (position == -1 and closes[t] >= short_stop)
                            or t - entered >= p.horizon):
                        position = 0
            elif entry[t] and np.isfinite(closes[t]) and closes[t] > 0:
                position, entered = int(entry[t]), t
                high, low, stop, short_stop = np.nan, np.nan, -np.inf, np.inf
            signals[t] = position
        df = pd.DataFrame({"signal": signals, "size": 0.95}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


if __name__ == "__main__":
    p = DrawdownAnalogParams(history=126, neighbors=8, threshold_window=32)
    strategy = GeneratedStrategy.__new__(GeneratedStrategy)
    rng = np.random.default_rng(17)
    close = 100 * np.exp(np.cumsum(rng.normal(0, 0.012, 500)))
    data = pd.DataFrame(
        {"open": close, "high": close * 1.01, "low": close * 0.99,
         "close": close, "volume": 1000000.0},
        index=pd.date_range("2020-01-01", periods=len(close), freq="B")
    )
    full = strategy.indicators(data, p)
    prefix = strategy.indicators(data.iloc[:420], p)
    pd.testing.assert_frame_equal(full.iloc[:420], prefix)
    result = strategy.generate_signals(data, full, None, p).data
    assert result["signal"].isin([-1, 0, 1]).all() and (result["size"] > 0).all()
    assert (result["signal"] != 0).any()
    pd.testing.assert_frame_equal(
        result.iloc[:420], strategy.generate_signals(data.iloc[:420], prefix, None, p).data
    )
    probe = data.iloc[:6].copy()
    probe["close"] = [100.0, 100.0, 105.0, 103.0, 99.0, 99.0]
    forced = pd.DataFrame(
        {"atr": [1.0, 1.0, 1.0, 5.0, 5.0, 5.0], "depth": 1.0,
         "depth_threshold": 0.5, "score": [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
         "score_threshold": 0.5}, index=probe.index
    )
    check = strategy.generate_signals(probe, forced, None, p).data["signal"]
    assert check.tolist() == [0, 1, 1, 1, 1, 0]
    forced["score"] = [-1.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    probe["close"] = [100.0, 100.0, 99.0, 102.0, 102.0, 102.0]
    forced["atr"] = 1.0
    check = strategy.generate_signals(probe, forced, None, p).data["signal"]
    assert check.tolist() == [0, -1, -1, -1, 0, 0]
