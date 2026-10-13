from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class StreakAnalogParams:
    shape_bars: int = 6
    history_bars: int = 252
    analog_percentile: float = 0.25
    forecast_percentile: float = 0.65
    min_analogs: int = 5
    hold_bars: int = 8
    profit_target: float = 0.025
    size: float = 0.95


def _position_path(opens, closes, entries, profit_target, hold_bars):
    decisions = np.zeros(len(closes), dtype=int)
    active = False
    entry_price = np.nan
    entry_bar = -1
    for i in range(len(closes)):
        if not active and i > 0 and decisions[i - 1] == 1:
            if not np.isfinite(opens[i]) or opens[i] <= 0:
                continue
            active = True
            entry_price = opens[i]
            entry_bar = i
        if active:
            target_hit = np.isfinite(closes[i]) and closes[i] >= entry_price * (1.0 + profit_target)
            if target_hit or i - entry_bar + 1 >= hold_bars:
                active = False
                continue
            decisions[i] = 1
        elif entries[i]:
            decisions[i] = 1
    return decisions


class GeneratedStrategy(BaseStrategy[StreakAnalogParams]):
    strategy_id = "gen_local-w2_1791215621"

    @classmethod
    def params_type(cls):
        return StreakAnalogParams

    def warmup_bars(self, params):
        if not 2 <= params.shape_bars <= params.history_bars:
            raise ValueError("Require 2 <= shape_bars <= history_bars")
        if not 0.0 < params.analog_percentile <= 1.0 or not 0.0 < params.forecast_percentile < 1.0:
            raise ValueError("Invalid percentile")
        if params.min_analogs < 2 or not 5 <= params.hold_bars <= 10:
            raise ValueError("Require min_analogs >= 2 and 5 <= hold_bars <= 10")
        if not np.isfinite(params.profit_target) or params.profit_target <= 0:
            raise ValueError("profit_target must be positive")
        if not np.isfinite(params.size) or not 0 < params.size <= 1:
            raise ValueError("size must be in (0, 1]")
        return int(params.history_bars + params.hold_bars + params.shape_bars + 1)

    def indicators(self, data, params):
        warmup = self.warmup_bars(params)
        out = pd.DataFrame(index=data.index)
        close = data["close"].where(np.isfinite(data["close"]) & (data["close"] > 0))
        opens = data["open"].where(np.isfinite(data["open"]) & (data["open"] > 0))
        direction = np.sign(close.diff())
        groups = direction.fillna(0).ne(direction.fillna(0).shift()).cumsum()
        out["streak"] = (direction.groupby(groups).cumcount() + 1).clip(upper=params.history_bars) * direction
        out["reversal"] = (out["streak"].eq(1) & out["streak"].shift(1).lt(0)).astype(int)
        shape_columns = []
        for lag in range(params.shape_bars):
            name = "streak_lag_" + str(lag)
            out[name] = out["streak"].shift(lag)
            shape_columns.append(name)
        # Outcome is timestamped at its completion, never at its historical entry.
        out["completed_return"] = close / opens.shift(params.hold_bars - 1) - 1.0
        shapes = out[shape_columns].to_numpy(dtype=float)
        reversals = out["reversal"].to_numpy(dtype=bool)
        completed = out["completed_return"].to_numpy(dtype=float)
        forecast = np.full(len(data), np.nan)
        threshold = np.full(len(data), np.nan)
        distance_cutoff = np.full(len(data), np.nan)
        analog_count = np.zeros(len(data), dtype=int)
        # ponytail: O(bars * history_bars * shape_bars); index neighbors if history grows substantially.
        for i in range(warmup - 1, len(data)):
            if not reversals[i] or not np.isfinite(shapes[i]).all():
                continue
            end = i - params.hold_bars
            candidates = np.arange(end - params.history_bars, end)
            outcomes = completed[candidates + params.hold_bars]
            valid = reversals[candidates] & np.isfinite(shapes[candidates]).all(axis=1) & np.isfinite(outcomes)
            candidates = candidates[valid]
            outcomes = outcomes[valid]
            if len(candidates) < params.min_analogs:
                continue
            distances = np.abs(shapes[candidates] - shapes[i]).mean(axis=1)
            cutoff = np.quantile(distances, params.analog_percentile)
            selected = distances <= cutoff
            distance_cutoff[i] = cutoff
            analog_count[i] = int(selected.sum())
            if analog_count[i] < params.min_analogs:
                continue
            forecast[i] = float(outcomes[selected].mean())
            threshold[i] = float(np.quantile(outcomes, params.forecast_percentile))
        out["distance_cutoff"] = distance_cutoff
        out["analog_count"] = analog_count
        out["forecast"] = forecast
        out["forecast_threshold"] = threshold
        out["entry"] = (out["reversal"].eq(1) & out["forecast"].gt(out["forecast_threshold"]) & out["forecast"].gt(0)).astype(int)
        return out

    def generate_signals(self, data, indicators, ctx: StrategyContext, params):
        self.warmup_bars(params)
        entries = indicators["entry"].fillna(0).to_numpy(dtype=bool)
        decisions = _position_path(
            data["open"].to_numpy(dtype=float),
            data["close"].to_numpy(dtype=float),
            entries,
            params.profit_target,
            params.hold_bars,
        )
        df = pd.DataFrame({"signal": decisions, "size": float(params.size)}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check():
    opens = np.full(12, 100.0)
    entries = np.zeros(12, dtype=bool)
    entries[0] = True
    closes = opens.copy()
    closes[3] = 103.0
    target_path = _position_path(opens, closes, entries, 0.025, 8)
    assert target_path.tolist() == [1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    time_path = _position_path(opens, opens, entries, 0.025, 8)
    assert time_path.tolist() == [1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0]
    shifted = pd.Series(time_path).shift(1).fillna(0).astype(int)
    assert shifted.iloc[0] == 0 and shifted.iloc[1:9].eq(1).all() and shifted.iloc[9:].eq(0).all()
    changed = closes.copy()
    changed[6:] = 120.0
    assert np.array_equal(target_path[:6], _position_path(opens, changed, entries, 0.025, 8)[:6])


if __name__ == "__main__":
    _self_check()
