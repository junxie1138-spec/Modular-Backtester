from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class ShapeParams:
    memory_bars: int = 126
    trail_atr: float = 1.5

    def __post_init__(self):
        if not isinstance(self.memory_bars, int) or self.memory_bars < 30:
            raise ValueError("memory_bars must be an integer >= 30")
        if not np.isfinite(self.trail_atr) or self.trail_atr <= 0:
            raise ValueError("trail_atr must be finite and positive")


class GeneratedStrategy(BaseStrategy[ShapeParams]):
    strategy_id = "gen_local-w2_1791228153"

    @classmethod
    def params_type(cls):
        return ShapeParams

    def warmup_bars(self, params):
        return int(max(params.memory_bars + 5, 15))

    def indicators(self, data, params):
        out = pd.DataFrame(index=data.index)
        close = data["close"].where(data["close"] > 0)
        out["range_fraction"] = (data["high"] - data["low"]).clip(lower=0) / close
        scale = out["range_fraction"].rolling(5, min_periods=5).mean().replace(0, np.nan)
        for k in range(5):
            out[f"shape_{k}"] = out["range_fraction"].shift(k) / scale
        previous = close.shift(1)
        tr = pd.concat([
            data["high"] - data["low"],
            (data["high"] - previous).abs(),
            (data["low"] - previous).abs(),
        ], axis=1).max(axis=1)
        out["atr"] = tr.rolling(14, min_periods=14).mean()
        out["forecast"] = np.nan
        out["agreement"] = np.nan
        out["match_distance"] = np.nan
        out["entry_signal"] = 0
        if data.empty:
            return out
        features = out[[f"shape_{k}" for k in range(5)]]
        distances, outcomes = [], []
        # ponytail: O(bars * memory) scan; chunk rows if histories exhaust RAM.
        for lag in range(10, params.memory_bars + 1):
            delta = features - features.shift(lag)
            distance = np.square(delta).mean(axis=1, skipna=False)
            # Analog entry at j+1 open and exit at j+5 open are already observed.
            denominator = data["open"].shift(lag - 1).where(lambda x: x > 0)
            outcome = data["open"].shift(lag - 5) / denominator - 1.0
            valid = np.isfinite(distance) & np.isfinite(outcome)
            distances.append(distance.where(valid, np.inf).to_numpy())
            outcomes.append(outcome.to_numpy())
        distance_matrix = np.column_stack(distances)
        outcome_matrix = np.column_stack(outcomes)
        nearest = np.argpartition(distance_matrix, 6, axis=1)[:, :7]
        matched_distance = np.take_along_axis(distance_matrix, nearest, axis=1)
        matched_return = np.take_along_axis(outcome_matrix, nearest, axis=1)
        ready = np.isfinite(matched_distance).all(axis=1)
        rows = np.flatnonzero(ready)
        if len(rows):
            forecast = np.median(matched_return[rows], axis=1)
            agreement = (np.sign(matched_return[rows]) == np.sign(forecast[:, None])).mean(axis=1)
            out.iloc[rows, out.columns.get_loc("forecast")] = forecast
            out.iloc[rows, out.columns.get_loc("agreement")] = agreement
            out.iloc[rows, out.columns.get_loc("match_distance")] = matched_distance[rows].mean(axis=1)
        eligible = (
            (out["forecast"].abs() >= 0.0035)
            & (out["agreement"] >= 6.0 / 7.0)
            & (out["match_distance"] <= 0.12)
            & out["atr"].gt(0)
        )
        eligible.iloc[:self.warmup_bars(params)] = False
        out["entry_signal"] = np.sign(out["forecast"].fillna(0)).where(eligible, 0).astype(int)
        return out.replace([np.inf, -np.inf], np.nan)

    def generate_signals(self, data, indicators, ctx: StrategyContext, params):
        targets = np.zeros(len(data), dtype=int)
        opens = data["open"].to_numpy(dtype=float)
        highs = data["high"].to_numpy(dtype=float)
        lows = data["low"].to_numpy(dtype=float)
        closes = data["close"].to_numpy(dtype=float)
        atr = indicators["atr"].to_numpy(dtype=float)
        candidates = indicators["entry_signal"].fillna(0).to_numpy(dtype=int)
        position, pending, age = 0, 0, 0
        entry, peak, stop = np.nan, np.nan, np.nan
        armed = False
        for i in range(len(data)):
            valid = (
                np.isfinite([opens[i], highs[i], lows[i], closes[i], atr[i]]).all()
                and min(opens[i], highs[i], lows[i], closes[i], atr[i]) > 0
                and lows[i] <= min(opens[i], closes[i])
                and highs[i] >= max(opens[i], closes[i])
            )
            if not valid:
                position, pending, age = 0, 0, 0
                continue
            if pending:
                position, pending, age = pending, 0, 0
                entry = opens[i]
                peak = position * entry
                stop = peak - 2.0 * atr[i - 1]
                armed = False
            if position:
                age += 1
                adverse = lows[i] if position == 1 else -highs[i]
                favorable = highs[i] if position == 1 else -lows[i]
                # Inspect the existing stop before updating it from this bar.
                # All exits remain next-open instructions, including gap breaches.
                if adverse <= stop or age >= 4:
                    position = 0
                    continue
                peak = max(peak, favorable)
                if peak - position * entry >= 0.0075 * entry:
                    armed = True
                if armed:
                    # Signed stops only rise; actual short buy-stops only fall.
                    stop = max(stop, position * entry, peak - params.trail_atr * atr[i])
                targets[i] = position
            elif candidates[i] in (-1, 1):
                pending = int(candidates[i])
                targets[i] = pending
        df = pd.DataFrame({"signal": targets, "size": 1.0}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check():
    strategy = GeneratedStrategy()
    params = ShapeParams(memory_bars=30)
    index = pd.date_range("2020-01-01", periods=100)
    t = np.arange(100, dtype=float)
    close = 100.0 + 0.02 * t + 0.3 * np.sin(t)
    data = pd.DataFrame({
        "open": close, "high": close + 0.6 + 0.2 * np.cos(t),
        "low": close - 0.6 - 0.2 * np.cos(t), "close": close,
        "volume": 1000000.0,
    }, index=index)
    full = strategy.indicators(data, params)
    prefix = strategy.indicators(data.iloc[:80], params)
    pd.testing.assert_frame_equal(full.iloc[:80], prefix)
    for direction in (-1, 1):
        bars = pd.DataFrame({
            "open": [100.0] * 6, "high": [100.2] * 6,
            "low": [99.8] * 6, "close": [100.0] * 6,
            "volume": [1000000.0] * 6,
        }, index=index[:6])
        bars.iloc[1, bars.columns.get_loc("high" if direction == 1 else "low")] = 101.0 if direction == 1 else 99.0
        ind = pd.DataFrame({"atr": 1.0, "entry_signal": [direction, 0, 0, 0, 0, 0]}, index=bars.index)
        result = strategy.generate_signals(bars, ind, None, params).data
        assert result["signal"].tolist() == [0, direction, direction, 0, 0, 0]
        assert result["size"].gt(0).all()
        bars["high"], bars["low"] = 100.2, 99.8
        timed = strategy.generate_signals(bars, ind, None, params).data
        assert timed["signal"].tolist() == [0, direction, direction, direction, direction, 0]


if __name__ == "__main__":
    _self_check()
