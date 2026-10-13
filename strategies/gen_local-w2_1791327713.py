from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class RangeAnalogParams:
    shape_bars: int = 4
    history_bars: int = 252
    analog_percentile: float = 0.15
    min_analogs: int = 12
    atr_bars: int = 14
    stop_atr: float = 1.25
    hold_bars: int = 2

    def __post_init__(self):
        if not (2 <= self.shape_bars and self.history_bars > self.shape_bars + self.hold_bars + self.min_analogs):
            raise ValueError("Insufficient analog history")
        if not (0 < self.analog_percentile < 1 and self.min_analogs >= 1):
            raise ValueError("Invalid analog selection")
        if not (self.atr_bars >= 1 and np.isfinite(self.stop_atr) and self.stop_atr > 0 and self.hold_bars in (1, 2)):
            raise ValueError("Invalid risk parameters")


class GeneratedStrategy(BaseStrategy[RangeAnalogParams]):
    strategy_id = "gen_local-w2_1791327713"

    @classmethod
    def params_type(cls):
        return RangeAnalogParams

    def warmup_bars(self, params):
        return int(max(params.history_bars + params.shape_bars + 1, params.atr_bars + 1))

    def indicators(self, data, params):
        out = pd.DataFrame(index=data.index)
        previous = data["close"].shift(1).replace(0, np.nan)
        out["range_fraction"] = (data["high"] - data["low"]).clip(lower=0).div(previous)
        out["range_scale"] = out["range_fraction"].rolling(params.shape_bars).mean().replace(0, np.nan)
        features = []
        for k in range(params.shape_bars):
            name = f"shape_{k}"
            out[name] = out["range_fraction"].shift(k).div(out["range_scale"])
            features.append(name)
        out["true_range"] = pd.concat([
            data["high"] - data["low"],
            (data["high"] - previous).abs(),
            (data["low"] - previous).abs(),
        ], axis=1).max(axis=1)
        out["atr"] = out["true_range"].rolling(params.atr_bars).mean()
        if data.empty:
            for name in ("distance_cutoff", "analog_count", "forecast"):
                out[name] = pd.Series(index=data.index, dtype=float)
            return out
        shape = out[features].replace([np.inf, -np.inf], np.nan)
        distances, outcomes = [], []
        # Candidate outcomes finish before the current shape begins.
        # ponytail: O(history * bars) storage; chunk lag matrices for much longer histories.
        for lag in range(params.shape_bars + params.hold_bars + 1, params.history_bars + 1):
            distances.append((shape - shape.shift(lag)).pow(2).mean(axis=1, skipna=False).to_numpy())
            outcomes.append((
                data["open"].shift(lag - params.hold_bars - 1)
                .div(data["open"].shift(lag - 1).replace(0, np.nan)) - 1
            ).to_numpy())
        distance = np.column_stack(distances)
        outcome = np.column_stack(outcomes)
        distance[~np.isfinite(outcome)] = np.nan
        cutoff = pd.DataFrame(distance).quantile(params.analog_percentile, axis=1).to_numpy()
        selected = np.isfinite(distance) & np.isfinite(outcome) & (distance <= cutoff[:, None])
        count = selected.sum(axis=1)
        forecast = np.where(selected, outcome, 0.0).sum(axis=1) / np.maximum(count, 1)
        forecast[count < params.min_analogs] = np.nan
        out["distance_cutoff"] = cutoff
        out["analog_count"] = count
        out["forecast"] = forecast
        return out.replace([np.inf, -np.inf], np.nan)

    def generate_signals(self, data, indicators, ctx, params):
        df = pd.DataFrame(index=data.index)
        raw = np.zeros(len(data), dtype=int)
        opening = data["open"].to_numpy(dtype=float)
        close = data["close"].to_numpy(dtype=float)
        atr = indicators["atr"].to_numpy(dtype=float)
        forecast = indicators["forecast"].to_numpy(dtype=float)
        position, age = 0, 0
        entry, stop_distance = np.nan, np.nan
        for i in range(len(data)):
            if position:
                # Latch the next-bar entry open only when that bar is observed.
                if age == 0:
                    entry = opening[i]
                age += 1
                valid = np.isfinite(entry) and entry > 0 and np.isfinite(close[i])
                stopped = valid and position * (close[i] - entry) < -stop_distance
                if not valid or stopped or age >= params.hold_bars:
                    position = 0
            elif i >= self.warmup_bars(params) - 1:
                if np.isfinite(forecast[i]) and forecast[i] != 0 and np.isfinite(atr[i]) and atr[i] > 0:
                    position = 1 if forecast[i] > 0 else -1
                    stop_distance = params.stop_atr * atr[i]
                    entry, age = np.nan, 0
            raw[i] = position
        df["signal"] = raw
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = 0.95
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check():
    params = RangeAnalogParams(history_bars=30, min_analogs=2, atr_bars=3)
    n = 100
    x = np.arange(n, dtype=float)
    close = 100 + 0.03 * x + np.sin(x * 0.8)
    opening = close + 0.2 * np.cos(x)
    data = pd.DataFrame({
        "open": opening, "high": np.maximum(opening, close) + 1 + 0.3 * np.sin(x),
        "low": np.minimum(opening, close) - 1, "close": close, "volume": 1000.0,
    }, index=pd.date_range("2020-01-01", periods=n))
    strategy = GeneratedStrategy()
    full = strategy.indicators(data, params)
    prefix = strategy.indicators(data.iloc[:80], params)
    assert np.allclose(full.iloc[:80], prefix, equal_nan=True)
    a = strategy.generate_signals(data, full, None, params).data
    b = strategy.generate_signals(data.iloc[:80], prefix, None, params).data
    assert a.iloc[:80].equals(b)
    assert a["signal"].isin([-1, 0, 1]).all() and (a["size"] > 0).all()
    start = strategy.warmup_bars(params) - 1
    for side in (1, -1):
        fixture = data.copy()
        fixture["open"] = 100.0
        fixture["close"] = 100.0
        ind = full.copy()
        ind["forecast"] = np.nan
        ind["atr"] = 2.0
        ind.iloc[start, ind.columns.get_loc("forecast")] = side
        fixture.iloc[start + 1, fixture.columns.get_loc("close")] = 100 - side * 3
        result = strategy.generate_signals(fixture, ind, None, params).data
        assert result["signal"].iloc[start] == 0
        assert result["signal"].iloc[start + 1] == side
        assert result["signal"].iloc[start + 2] == 0
        fixture["close"] = 100.0
        result = strategy.generate_signals(fixture, ind, None, params).data
        assert result["signal"].iloc[start + 1:start + 4].tolist() == [side, side, 0]


if __name__ == "__main__":
    _self_check()
