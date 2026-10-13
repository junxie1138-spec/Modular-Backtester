from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class Params:
    frac_window: int = 32
    frac_order: float = 0.45
    trend_window: int = 3
    atr_window: int = 14
    frac_threshold: float = 0.25
    migration_threshold: float = 0.12
    trail_atr: float = 1.8
    hold_bars: int = 4
    size: float = 0.95


class GeneratedStrategy(BaseStrategy[Params]):
    strategy_id = "gen_local_1790888561"

    @classmethod
    def params_type(cls):
        return Params

    def warmup_bars(self, params: Params) -> int:
        return int(max(params.frac_window + params.trend_window,
                       params.atr_window + 1, params.trend_window + 1))

    def indicators(self, data: pd.DataFrame, params: Params) -> pd.DataFrame:
        if not (params.frac_window >= 2 and params.trend_window >= 1
                and params.atr_window >= 1 and 0 < params.frac_order < 1
                and params.frac_threshold > 0 and params.migration_threshold > 0
                and params.trail_atr > 0 and 3 <= params.hold_bars <= 5
                and 0 < params.size <= 1):
            raise ValueError("Invalid strategy parameters")
        out = pd.DataFrame(index=data.index)
        close = data["close"].astype(float).replace([np.inf, -np.inf], np.nan)
        out["previous_close"] = close.shift(1)
        out["true_range"] = pd.concat([
            data["high"] - data["low"],
            (data["high"] - out["previous_close"]).abs(),
            (data["low"] - out["previous_close"]).abs(),
        ], axis=1).max(axis=1, skipna=False)
        out["atr"] = out["true_range"].rolling(params.atr_window).mean()
        weights = np.empty(params.frac_window, dtype=float)
        weights[0] = 1.0
        for lag in range(1, params.frac_window):
            weights[lag] = -weights[lag - 1] * (params.frac_order - lag + 1) / lag
        # ponytail: fixed finite-memory truncation; lengthen frac_window for longer memory.
        fractional = pd.Series(0.0, index=data.index)
        for lag, weight in enumerate(weights):
            fractional = fractional + weight * close.shift(lag)
        out["fractional_close"] = fractional
        out["fractional_momentum"] = fractional.diff(params.trend_window) / out["atr"].where(out["atr"] > 0)
        out["midpoint"] = (data["high"] + data["low"]) / 2.0
        out["midpoint_change"] = out["midpoint"].diff()
        out["range_budget"] = out["true_range"].rolling(params.trend_window).sum()
        out["midpoint_migration"] = (
            out["midpoint_change"].rolling(params.trend_window).sum()
            / out["range_budget"].where(out["range_budget"] > 0)
        )
        return out.replace([np.inf, -np.inf], np.nan)

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame,
                         ctx: StrategyContext, params: Params) -> SignalFrame:
        momentum = indicators["fractional_momentum"]
        migration = indicators["midpoint_migration"]
        entries = np.select([
            (momentum > params.frac_threshold) & (migration > params.migration_threshold),
            (momentum < -params.frac_threshold) & (migration < -params.migration_threshold),
        ], [1, -1], default=0)
        closes = data["close"].to_numpy(dtype=float)
        atrs = indicators["atr"].to_numpy(dtype=float)
        ready = (
            np.isfinite(closes) & (closes > 0) & np.isfinite(atrs) & (atrs > 0)
            & np.isfinite(momentum.to_numpy(dtype=float))
            & np.isfinite(migration.to_numpy(dtype=float))
        )
        raw = np.zeros(len(data), dtype=int)
        position, age = 0, 0
        extreme, stop = np.nan, np.nan
        for i in range(len(data)):
            if not ready[i]:
                position, age = 0, 0
                extreme, stop = np.nan, np.nan
            elif position:
                # The pending entry filled at today's open; track closes from today.
                age += 1
                if position == 1:
                    extreme = closes[i] if np.isnan(extreme) else max(extreme, closes[i])
                    candidate = extreme - params.trail_atr * atrs[i]
                    stop = candidate if np.isnan(stop) else max(stop, candidate)
                    stopped = closes[i] <= stop
                else:
                    extreme = closes[i] if np.isnan(extreme) else min(extreme, closes[i])
                    candidate = extreme + params.trail_atr * atrs[i]
                    stop = candidate if np.isnan(stop) else min(stop, candidate)
                    stopped = closes[i] >= stop
                if stopped or age >= params.hold_bars:
                    position, age = 0, 0
                    extreme, stop = np.nan, np.nan
            elif entries[i]:
                position, age = int(entries[i]), 0
                extreme, stop = np.nan, np.nan
            raw[i] = position
        df = pd.DataFrame({"signal": raw, "size": float(params.size)}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


if __name__ == "__main__":
    index = pd.date_range("2020-01-01", periods=12)
    close = pd.Series([100, 100, 101, 104, 102, 101, 100, 98, 96, 98, 99, 100], index=index, dtype=float)
    data = pd.DataFrame({"open": close, "high": close + 0.5,
                         "low": close - 0.5, "close": close, "volume": 1000.0})
    indicators = pd.DataFrame({
        "fractional_momentum": [1, 1, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0],
        "midpoint_migration": [-1, 1, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0],
        "atr": 1.0,
    }, index=index)
    strategy = GeneratedStrategy()
    params = Params(trail_atr=1.5)
    ctx = StrategyContext(symbol="SPY", timeframe="1d", warmup_bars=strategy.warmup_bars(params))
    result = strategy.generate_signals(data, indicators, ctx, params).data
    assert result["signal"].tolist() == [0, 0, 1, 1, 1, 0, 0, -1, -1, -1, 0, 0]
    assert result["size"].gt(0).all()
    flat = pd.DataFrame({"open": 100.0, "high": 101.0, "low": 99.0,
                         "close": 100.0, "volume": 1000.0},
                        index=pd.date_range("2020-01-01", periods=80))
    derived = strategy.indicators(flat, params)
    assert derived["fractional_momentum"].iloc[:params.frac_window + params.trend_window - 1].isna().all()
    assert strategy.generate_signals(flat, derived, ctx, params).data["signal"].eq(0).all()
