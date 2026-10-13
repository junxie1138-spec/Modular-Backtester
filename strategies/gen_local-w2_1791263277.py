from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class Params:
    frac_order: float = 0.45
    frac_window: int = 60
    breakout_window: int = 12
    compression_short: int = 8
    compression_long: int = 40
    compression_threshold: float = 0.42
    profit_target: float = 0.035
    max_hold_bars: int = 18


class GeneratedStrategy(BaseStrategy[Params]):
    strategy_id = "gen_local-w2_1791263277"

    @classmethod
    def params_type(cls) -> type[Params]:
        return Params

    @staticmethod
    def warmup_bars(params: Params) -> int:
        if not (0.0 < params.frac_order < 1.0):
            raise ValueError("frac_order must be between zero and one")
        if not (2 <= params.compression_short < params.compression_long):
            raise ValueError("compression windows must satisfy 2 <= short < long")
        if params.frac_window < 2 or params.breakout_window < 1:
            raise ValueError("fractional and breakout windows must be positive")
        if not (0.0 < params.compression_threshold < 1.0):
            raise ValueError("compression_threshold must be between zero and one")
        if params.profit_target <= 0.0 or params.max_hold_bars < 1:
            raise ValueError("profit_target and max_hold_bars must be positive")
        return int(max(params.frac_window + params.breakout_window,
                       params.compression_long))

    @staticmethod
    def indicators(data: pd.DataFrame, params: Params) -> pd.DataFrame:
        GeneratedStrategy.warmup_bars(params)
        out = pd.DataFrame(index=data.index)
        out["log_close"] = np.log(data["close"].where(data["close"] > 0.0))
        fractional = out["log_close"].copy()
        weight = 1.0
        for lag in range(1, params.frac_window):
            weight *= (lag - 1.0 - params.frac_order) / lag
            fractional = fractional + weight * out["log_close"].shift(lag)
        out["fractional_close"] = fractional
        out["fractional_ceiling"] = fractional.shift(1).rolling(
            params.breakout_window, min_periods=params.breakout_window
        ).max()
        out["short_span"] = (
            data["high"].rolling(params.compression_short).max()
            - data["low"].rolling(params.compression_short).min()
        )
        out["long_span"] = (
            data["high"].rolling(params.compression_long).max()
            - data["low"].rolling(params.compression_long).min()
        )
        out["compression_ratio"] = (
            out["short_span"] / out["long_span"].where(out["long_span"] > 0.0)
        ).replace([np.inf, -np.inf], np.nan)
        return out

    @staticmethod
    def generate_signals(data: pd.DataFrame, indicators: pd.DataFrame,
                         ctx: StrategyContext, params: Params) -> SignalFrame:
        GeneratedStrategy.warmup_bars(params)
        ready = np.isfinite(indicators[["fractional_close", "fractional_ceiling",
                                       "compression_ratio"]]).all(axis=1)
        entry = (
            ready
            & (indicators["fractional_close"] > indicators["fractional_ceiling"])
            & (indicators["compression_ratio"] <= params.compression_threshold)
        ).fillna(False).to_numpy(dtype=bool)
        opens = data["open"].to_numpy(dtype=float)
        closes = data["close"].to_numpy(dtype=float)
        desired = np.zeros(len(data), dtype=int)
        entry_bar = -1
        entry_price = np.nan
        for i in range(len(data)):
            # Yesterday's entry decision fills at today's open.
            if entry_bar < 0 and i > 0 and desired[i - 1] == 1:
                entry_bar = i
                entry_price = opens[i]
            if entry_bar >= 0:
                # Evaluate the gross profit target at the close; exit next open.
                target_hit = (np.isfinite(closes[i])
                              and closes[i] >= entry_price * (1.0 + params.profit_target))
                time_up = i - entry_bar + 1 >= params.max_hold_bars
                if target_hit or time_up:
                    entry_bar = -1
                    entry_price = np.nan
                else:
                    desired[i] = 1
            elif entry[i]:
                desired[i] = 1
        df = pd.DataFrame({"signal": desired, "size": 0.95}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check() -> None:
    index = pd.date_range("2020-01-01", periods=7)
    data = pd.DataFrame({"open": 100.0, "high": 105.0, "low": 99.0,
                         "close": 100.0, "volume": 1000.0}, index=index)
    data.iloc[2, data.columns.get_loc("close")] = 104.0
    indicators = pd.DataFrame({"fractional_close": [0, 2, 0, 0, 0, 0, 0],
                               "fractional_ceiling": 1.0,
                               "compression_ratio": 0.3}, index=index)
    profit = GeneratedStrategy.generate_signals(data, indicators, None, Params()).data
    assert profit["signal"].tolist() == [0, 0, 1, 0, 0, 0, 0]
    timed = GeneratedStrategy.generate_signals(
        data, indicators, None, Params(profit_target=1.0, max_hold_bars=3)
    ).data
    assert timed["signal"].tolist() == [0, 0, 1, 1, 1, 0, 0]
    blocked = indicators.copy()
    blocked["compression_ratio"] = 0.9
    assert not GeneratedStrategy.generate_signals(data, blocked, None, Params()).data["signal"].any()


if __name__ == "__main__":
    _self_check()
