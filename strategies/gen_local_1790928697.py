from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class RatchetParams:
    peak_window: int = 20
    pressure_window: int = 4
    min_drawdown: float = 0.015
    min_impulse: float = 0.15
    max_undo_fraction: float = 0.5


class GeneratedStrategy(BaseStrategy[RatchetParams]):
    strategy_id = "gen_local_1790928697"

    @classmethod
    def params_type(cls):
        return RatchetParams

    def warmup_bars(self, params: RatchetParams) -> int:
        return int(params.peak_window + params.pressure_window + 1)

    def indicators(self, data: pd.DataFrame, params: RatchetParams) -> pd.DataFrame:
        if params.peak_window < 2 or params.pressure_window < 2:
            raise ValueError("Lookback windows must be at least two bars.")
        if not np.isfinite(params.min_drawdown) or params.min_drawdown <= 0:
            raise ValueError("min_drawdown must be positive and finite.")
        if not np.isfinite(params.min_impulse) or params.min_impulse <= 0:
            raise ValueError("min_impulse must be positive and finite.")
        if not np.isfinite(params.max_undo_fraction) or not 0 <= params.max_undo_fraction <= 1:
            raise ValueError("max_undo_fraction must be between zero and one.")
        out = pd.DataFrame(index=data.index)
        out["prior_close"] = data["close"].shift(1)
        out["peak"] = data["close"].rolling(params.peak_window, min_periods=params.peak_window).max().shift(1)
        denominator = out["peak"].where(out["peak"] > 0)
        out["drawdown"] = (1.0 - out["prior_close"] / denominator).clip(lower=0.0)
        out["gap_release"] = (data["open"] - out["prior_close"]) / denominator
        out["intraday_release"] = (data["close"] - data["open"]) / denominator
        window = params.pressure_window
        out["gap_sum"] = out["gap_release"].rolling(window, min_periods=window).sum()
        out["intraday_sum"] = out["intraday_release"].rolling(window, min_periods=window).sum()
        out["depth"] = out["drawdown"].rolling(window, min_periods=window).mean()
        out["impulse"] = out["gap_sum"] / out["depth"].clip(lower=params.min_drawdown)
        return out.replace([np.inf, -np.inf], np.nan)

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: RatchetParams) -> SignalFrame:
        valid = indicators[["depth", "impulse", "gap_sum", "intraday_sum"]].notna().all(axis=1)
        deep = indicators["depth"].ge(params.min_drawdown)
        allowance = indicators["gap_sum"].abs() * params.max_undo_fraction
        long_entry = valid & deep & indicators["impulse"].ge(params.min_impulse) & indicators["intraday_sum"].ge(-allowance)
        short_entry = valid & deep & indicators["impulse"].le(-params.min_impulse) & indicators["intraday_sum"].le(allowance)
        candidate = pd.Series(np.select([long_entry, short_entry], [1, -1], default=0), index=data.index, dtype=int)
        confirmed = candidate.where(candidate.eq(candidate.shift(1)), 0).to_numpy(dtype=int)
        positions = np.zeros(len(data), dtype=int)
        position = 0
        for i, direction in enumerate(confirmed):
            if position == 0:
                position = int(direction)
            elif direction == -position:
                position = 0
            positions[i] = position
        df = pd.DataFrame({"signal": positions, "size": 0.95}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check():
    strategy = GeneratedStrategy()
    params = RatchetParams()
    index = pd.date_range("2020-01-01", periods=10)
    data = pd.DataFrame(100.0, index=index, columns=["open", "high", "low", "close", "volume"])
    directions = np.array([1, 1, 0, -1, -1, -1, 0, 1, 1, 1])
    indicators = pd.DataFrame({"depth": 0.03, "impulse": directions * 0.3, "gap_sum": directions * 0.009, "intraday_sum": 0.0}, index=index)
    ctx = StrategyContext(symbol="SPY", timeframe="1d", warmup_bars=strategy.warmup_bars(params))
    result = strategy.generate_signals(data, indicators, ctx, params).data
    assert result["signal"].tolist() == [0, 0, 1, 1, 1, 0, -1, -1, -1, 0]
    assert result["size"].gt(0).all()
    flat_index = pd.date_range("2020-01-01", periods=60)
    flat = pd.DataFrame(100.0, index=flat_index, columns=data.columns)
    flat_indicators = strategy.indicators(flat, params)
    assert strategy.generate_signals(flat, flat_indicators, ctx, params).data["signal"].eq(0).all()


if __name__ == "__main__":
    _self_check()
