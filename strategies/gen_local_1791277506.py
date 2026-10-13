from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class InventoryParams:
    peak_window: int = 63
    inventory_window: int = 8
    atr_window: int = 14
    min_depth: float = 0.012
    imbalance_threshold: float = 0.35
    breakeven_pct: float = 0.004
    trail_atr: float = 1.5
    hold_days: int = 2
    min_size: float = 0.25
    max_size: float = 0.95


class GeneratedStrategy(BaseStrategy[InventoryParams]):
    strategy_id = "gen_local_1791277506"

    @classmethod
    def params_type(cls):
        return InventoryParams

    def warmup_bars(self, params):
        return int(max(params.peak_window + params.inventory_window - 1,
                       params.peak_window + 1, params.atr_window + 1))

    def indicators(self, data, params):
        if min(params.peak_window, params.inventory_window, params.atr_window) < 1:
            raise ValueError("Indicator windows must be positive")
        close = data["close"]
        peak = close.rolling(params.peak_window, min_periods=params.peak_window).max()
        depth = (1.0 - close / peak.where(peak > 0)).clip(lower=0.0)
        volume = data["volume"].clip(lower=0.0)
        weight = depth * volume
        denominator = weight.rolling(params.inventory_window, min_periods=params.inventory_window).sum()
        numerator = (depth * weight).rolling(params.inventory_window, min_periods=params.inventory_window).sum()
        centroid = numerator / denominator.where(denominator > 0)
        previous_close = close.shift(1)
        true_range = pd.concat([
            data["high"] - data["low"],
            (data["high"] - previous_close).abs(),
            (data["low"] - previous_close).abs(),
        ], axis=1).max(axis=1)
        atr = true_range.rolling(params.atr_window, min_periods=params.atr_window).mean()
        atr_fraction = atr / close.where(close > 0)
        score = (centroid - depth) / atr_fraction.where(atr_fraction > 0)
        return pd.DataFrame({
            "peak": peak,
            "depth": depth,
            "inventory_weight": weight,
            "inventory_weight_sum": denominator,
            "inventory_depth_sum": numerator,
            "centroid": centroid,
            "previous_close": previous_close,
            "true_range": true_range,
            "atr": atr,
            "atr_fraction": atr_fraction,
            "score": score,
            "depth_change": depth.diff(),
        }, index=data.index).replace([np.inf, -np.inf], np.nan)

    def generate_signals(self, data, indicators, ctx, params):
        if not (params.imbalance_threshold > 0 and params.trail_atr > 0
                and params.breakeven_pct > 0 and params.min_depth >= 0
                and 1 <= params.hold_days <= 2
                and 0 < params.min_size <= params.max_size <= 1):
            raise ValueError("Invalid trading parameters")
        n = len(data)
        signals = np.zeros(n, dtype=int)
        sizes = np.ones(n, dtype=float)
        opens = data["open"].to_numpy(dtype=float)
        closes = data["close"].to_numpy(dtype=float)
        values = indicators[["depth", "depth_change", "score", "atr"]].to_numpy(dtype=float)
        side = 0
        entry = np.nan
        stop_profit = 0.0
        best_profit = 0.0
        armed = False
        age = 0
        active_size = 1.0
        for i in range(n):
            depth, change, score, atr = values[i]
            valid = (np.isfinite(values[i]).all() and np.isfinite(closes[i])
                     and closes[i] > 0 and atr > 0)
            exited = False
            if side:
                if np.isnan(entry):
                    # A previous close's decision fills at this bar's open.
                    entry = opens[i]
                    prior_atr = values[i - 1, 3]
                    stop_profit = -params.trail_atr * prior_atr
                    best_profit = 0.0
                    armed = False
                    age = 0
                if not valid or not np.isfinite(entry) or entry <= 0:
                    side = 0
                    entry = np.nan
                    exited = True
                else:
                    age += 1
                    profit = side * (closes[i] - entry)
                    best_profit = max(best_profit, profit)
                    armed = armed or profit >= entry * params.breakeven_pct
                    if armed:
                        # Signed profit floor only rises; short price stops fall.
                        stop_profit = max(stop_profit, 0.0,
                                          best_profit - params.trail_atr * atr)
                    # Stops are evaluated at the close and execute next open.
                    if profit <= stop_profit or age >= params.hold_days:
                        side = 0
                        entry = np.nan
                        exited = True
            if side == 0 and not exited and valid and depth >= params.min_depth:
                if score >= params.imbalance_threshold and change < 0:
                    side = 1
                elif score <= -params.imbalance_threshold and change > 0:
                    side = -1
                if side:
                    entry = np.nan
                    strength = min(abs(score) / (2.0 * params.imbalance_threshold), 1.0)
                    active_size = float(params.min_size +
                                        (params.max_size - params.min_size) * strength)
            signals[i] = side
            sizes[i] = active_size if side else 1.0
        df = pd.DataFrame({"signal": signals, "size": sizes}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = df["size"].shift(1).fillna(1.0).astype(float)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check():
    strategy = GeneratedStrategy()
    params = InventoryParams()
    index = pd.date_range("2020-01-01", periods=3)
    for side in (1, -1):
        close = np.array([100.0, 100.0 - 3.0 * side, 100.0])
        data = pd.DataFrame({"open": 100.0, "high": np.maximum(close, 100.0),
                             "low": np.minimum(close, 100.0), "close": close,
                             "volume": 1000.0}, index=index)
        indicators = pd.DataFrame({"depth": 0.03, "depth_change": -0.001 * side,
                                   "score": float(side), "atr": 1.0}, index=index)
        result = strategy.generate_signals(data, indicators, None, params).data
        assert result["signal"].tolist() == [0, side, 0]
        assert np.isfinite(result["size"]).all() and (result["size"] > 0).all()
    assert strategy.warmup_bars(params) >= params.peak_window + params.inventory_window - 1


if __name__ == "__main__":
    _self_check()
