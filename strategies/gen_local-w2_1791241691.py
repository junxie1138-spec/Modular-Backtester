from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class GeneratedParams:
    lookback: int = 20
    recovery_level: float = 0.55
    close_location_min: float = 0.75
    profit_target: float = 0.012
    max_hold: int = 4
    size: float = 0.95


class GeneratedStrategy(BaseStrategy[GeneratedParams]):
    strategy_id = "gen_local-w2_1791241691"

    @classmethod
    def params_type(cls):
        return GeneratedParams

    def warmup_bars(self, params: GeneratedParams) -> int:
        return int(params.lookback + 2)

    def indicators(self, data: pd.DataFrame, params: GeneratedParams) -> pd.DataFrame:
        if params.lookback < 2 or params.max_hold < 1:
            raise ValueError("lookback must be >= 2 and max_hold >= 1")
        values = [params.recovery_level, params.close_location_min,
                  params.profit_target, params.size]
        if not np.isfinite(values).all():
            raise ValueError("parameters must be finite")
        if not (0 < params.recovery_level < 1 and 0 < params.close_location_min <= 1
                and params.profit_target > 0 and params.size > 0):
            raise ValueError("invalid thresholds, target, or size")
        out = pd.DataFrame(index=data.index)
        out["return"] = data["close"].pct_change(fill_method=None).replace(
            [np.inf, -np.inf], np.nan)
        out["return_low"] = out["return"].rolling(params.lookback).min()
        out["return_high"] = out["return"].rolling(params.lookback).max()
        out["return_span"] = out["return_high"] - out["return_low"]
        out["return_position"] = (out["return"] - out["return_low"]) / out[
            "return_span"].where(out["return_span"] > 0)
        out["previous_return_position"] = out["return_position"].shift(1)
        out["daily_range"] = data["high"] - data["low"]
        out["close_location"] = (data["close"] - data["low"]) / out[
            "daily_range"].where(out["daily_range"] > 0)
        return out.replace([np.inf, -np.inf], np.nan)

    def generate_signals(
        self, data: pd.DataFrame, indicators: pd.DataFrame,
        ctx: StrategyContext, params: GeneratedParams,
    ) -> SignalFrame:
        recovering = (
            (indicators["return"] < 0)
            & (indicators["return_position"] >= params.recovery_level)
            & (indicators["previous_return_position"] < params.recovery_level)
        )
        accepted = indicators["close_location"] >= params.close_location_min
        eligible = (recovering & accepted).fillna(False).to_numpy(dtype=bool)
        opens = data["open"].to_numpy(dtype=float)
        closes = data["close"].to_numpy(dtype=float)
        raw = np.zeros(len(data), dtype=int)
        decision_bar = -1
        entry_price = np.nan
        for i in range(len(data)):
            if decision_bar >= 0:
                # Read the fill-bar open only when that bar has arrived.
                if i == decision_bar + 1:
                    entry_price = opens[i]
                invalid_fill = not np.isfinite(entry_price) or entry_price <= 0
                target_hit = (
                    np.isfinite(closes[i])
                    and closes[i] >= entry_price * (1 + params.profit_target)
                )
                timed_out = i - decision_bar >= params.max_hold
                if invalid_fill or target_hit or timed_out:
                    decision_bar = -1
                    entry_price = np.nan
                else:
                    raw[i] = 1
            elif eligible[i]:
                decision_bar = i
                raw[i] = 1
        df = pd.DataFrame(index=data.index)
        df["signal"] = raw
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = float(params.size)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


if __name__ == "__main__":
    strategy = GeneratedStrategy()
    params = GeneratedParams()
    index = pd.date_range("2020-01-01", periods=6)
    data = pd.DataFrame(
        {"open": 100.0, "high": 101.0, "low": 99.0,
         "close": 100.0, "volume": 1000.0}, index=index)
    ind = pd.DataFrame(
        {"return": -0.001, "return_position": 0.6,
         "previous_return_position": [0.4, 0.6, 0.6, 0.6, 0.6, 0.6],
         "close_location": 0.9}, index=index)
    ctx = StrategyContext("SPY", "1d", strategy.warmup_bars(params))
    def signals():
        return strategy.generate_signals(data, ind, ctx, params).data["signal"].tolist()
    assert signals() == [0, 1, 1, 1, 1, 0]
    data.loc[index[2], "close"] = 102.0
    assert signals() == [0, 1, 1, 0, 0, 0]
    ind.loc[index[0], "close_location"] = 0.5
    assert signals() == [0, 0, 0, 0, 0, 0]
    actual = strategy.indicators(data, params)
    assert strategy.generate_signals(data, actual, ctx, params).data["signal"].eq(0).all()
