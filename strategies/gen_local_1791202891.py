from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class Params:
    fractional_order: float = 0.45
    fractional_window: int = 32
    percentile_window: int = 126
    tail_percentile: float = 0.12
    efficiency_window: int = 8
    profit_target: float = 0.006
    max_hold_bars: int = 2


class GeneratedStrategy(BaseStrategy[Params]):
    strategy_id = "gen_local_1791202891"

    @classmethod
    def params_type(cls):
        return Params

    def warmup_bars(self, params: Params) -> int:
        return int(params.fractional_window + params.percentile_window + params.efficiency_window + 1)

    def indicators(self, data: pd.DataFrame, params: Params) -> pd.DataFrame:
        if not (
            0 < params.fractional_order < 1
            and params.fractional_window >= 2
            and params.percentile_window >= 10
            and 0 < params.tail_percentile < 0.5
            and params.efficiency_window >= 2
            and np.isfinite(params.profit_target)
            and params.profit_target > 0
            and 1 <= params.max_hold_bars <= 2
        ):
            raise ValueError("Invalid strategy parameters")
        close = data["close"].where(data["close"] > 0)
        out = pd.DataFrame(index=data.index)
        fd = close.copy()
        weight = 1.0
        for lag in range(1, params.fractional_window):
            weight *= (lag - 1 - params.fractional_order) / lag
            fd = fd + weight * close.shift(lag)
        out["fractional_close"] = fd
        out["fractional_scaled"] = fd / close
        history = out["fractional_scaled"].shift(1).rolling(params.percentile_window)
        out["lower_tail"] = history.quantile(params.tail_percentile)
        out["upper_tail"] = history.quantile(1 - params.tail_percentile)
        out["absolute_change"] = close.diff().abs()
        out["path_length"] = out["absolute_change"].rolling(params.efficiency_window).sum()
        out["displacement"] = close.diff(params.efficiency_window).abs()
        out["efficiency"] = out["displacement"] / out["path_length"].replace(0, np.nan)
        out["efficiency_limit"] = out["efficiency"].shift(1).rolling(params.percentile_window).quantile(0.5)
        out = out.replace([np.inf, -np.inf], np.nan)
        noisy = out["efficiency"] <= out["efficiency_limit"]
        lower = out["fractional_scaled"] < out["lower_tail"]
        upper = out["fractional_scaled"] > out["upper_tail"]
        fresh_lower = out["fractional_scaled"].shift(1) >= out["lower_tail"].shift(1)
        fresh_upper = out["fractional_scaled"].shift(1) <= out["upper_tail"].shift(1)
        out["entry"] = np.select(
            [noisy & lower & fresh_lower, noisy & upper & fresh_upper],
            [1, -1], default=0
        ).astype(int)
        return out

    def generate_signals(
        self, data: pd.DataFrame, indicators: pd.DataFrame,
        ctx: StrategyContext, params: Params
    ) -> SignalFrame:
        signals = np.zeros(len(data), dtype=int)
        entries = indicators["entry"].fillna(0).to_numpy()
        opens = data["open"].to_numpy(dtype=float)
        closes = data["close"].to_numpy(dtype=float)
        position = 0
        entry_price = np.nan
        entry_bar = -1
        for i in range(len(data)):
            if position:
                if entry_bar == -1:
                    # Yesterday's decision fills at today's open.
                    entry_price = opens[i]
                    entry_bar = i
                valid = np.isfinite(entry_price) and entry_price > 0 and np.isfinite(closes[i])
                gain = position * (closes[i] / entry_price - 1) if valid else -np.inf
                if not valid or gain >= params.profit_target or i - entry_bar + 1 >= params.max_hold_bars:
                    position = 0
                    entry_bar = -1
                    entry_price = np.nan
            elif entries[i] in (-1, 1) and np.isfinite(closes[i]) and closes[i] > 0:
                position = int(entries[i])
                entry_bar = -1
            signals[i] = position
        df = pd.DataFrame({"signal": signals, "size": 0.95}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


if __name__ == "__main__":
    strategy = GeneratedStrategy()
    params = Params()
    idx = pd.date_range("2020-01-01", periods=7)
    sample = pd.DataFrame({
        "open": 100.0, "high": 102.0, "low": 99.0,
        "close": [100.0, 101.0, 100.0, 100.0, 100.0, 100.0, 100.0],
        "volume": 1000.0
    }, index=idx)
    entries = pd.DataFrame({"entry": [1, 0, 0, -1, 0, 0, 0]}, index=idx)
    context = StrategyContext(symbol="SPY", timeframe="1d", warmup_bars=0)
    result = strategy.generate_signals(sample, entries, context, params).data
    assert result["signal"].tolist() == [0, 1, 0, 0, -1, -1, 0]
    assert (result["size"] > 0).all()
