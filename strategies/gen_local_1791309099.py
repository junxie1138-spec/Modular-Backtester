from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class GeneratedParams:
    rank_window: int = 63
    recent_bars: int = 7
    baseline_bars: int = 14
    entry_shift: float = 0.10
    min_size: float = 0.25
    max_size: float = 0.95


class GeneratedStrategy(BaseStrategy[GeneratedParams]):
    strategy_id = "gen_local_1791309099"

    @classmethod
    def params_type(cls):
        return GeneratedParams

    def warmup_bars(self, params: GeneratedParams) -> int:
        return int(params.rank_window + params.recent_bars + params.baseline_bars)

    def indicators(self, data: pd.DataFrame, params: GeneratedParams) -> pd.DataFrame:
        if min(params.rank_window, params.recent_bars, params.baseline_bars) < 1:
            raise ValueError("Lookbacks must be positive.")
        if not 0 <= params.entry_shift < 1 or not 0 < params.min_size <= params.max_size <= 1:
            raise ValueError("Invalid threshold or equity sizing bounds.")
        out = pd.DataFrame(index=data.index)
        out["range"] = (data["high"] - data["low"]).where(data["high"] >= data["low"])
        out["close_location"] = ((data["close"] - data["low"]) / out["range"].replace(0, np.nan)).replace([np.inf, -np.inf], np.nan).clip(0, 1)
        out.loc[out["range"].eq(0) & data["close"].notna(), "close_location"] = 0.5
        out["location_rank"] = out["close_location"].rolling(params.rank_window, min_periods=params.rank_window).rank(pct=True)
        out["recent_rank"] = out["location_rank"].rolling(params.recent_bars, min_periods=params.recent_bars).mean()
        out["baseline_rank"] = out["location_rank"].shift(params.recent_bars).rolling(params.baseline_bars, min_periods=params.baseline_bars).mean()
        out["rank_shift"] = out["recent_rank"] - out["baseline_rank"]
        out["conviction"] = ((out["rank_shift"] - params.entry_shift) / (1 - params.entry_shift)).clip(0, 1).fillna(0)
        return out

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: GeneratedParams,
    ) -> SignalFrame:
        df = pd.DataFrame(index=data.index)
        # The same condition maintains exposure and exits when it reverses.
        df["signal"] = indicators["rank_shift"].gt(params.entry_shift).astype(int)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        decision_size = params.min_size + (params.max_size - params.min_size) * indicators["conviction"]
        df["size"] = decision_size.shift(1).fillna(params.min_size).astype(float)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


if __name__ == "__main__":
    params = GeneratedParams()
    strategy = GeneratedStrategy()
    index = pd.date_range("2020-01-01", periods=220)
    location = 0.5 + 0.45 * np.sin(np.arange(len(index)) / 8)
    data = pd.DataFrame({"open": 100.5, "high": 101.0, "low": 100.0, "close": 100 + location, "volume": 1000.0}, index=index)
    ctx = StrategyContext(symbol="SPY", timeframe="1d", warmup_bars=strategy.warmup_bars(params))
    indicators = strategy.indicators(data, params)
    signals = strategy.generate_signals(data, indicators, ctx, params).data
    expected = indicators["rank_shift"].gt(params.entry_shift).shift(1).fillna(False).astype(int)
    assert signals["signal"].equals(expected)
    assert signals["signal"].eq(1).any() and signals["signal"].diff().eq(-1).any()
    assert signals["size"].between(params.min_size, params.max_size).all()
    assert signals.loc[signals["signal"].eq(1), "size"].nunique() > 1
    assert indicators.iloc[:150].equals(strategy.indicators(data.iloc[:150], params))
    flat = data.copy()
    flat[["open", "high", "low", "close"]] = 100.0
    assert strategy.generate_signals(flat, strategy.indicators(flat, params), ctx, params).data["signal"].eq(0).all()
