from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class AccelerationParams:
    roc_bars: int = 3
    participation_bars: int = 12
    acceleration_threshold: float = 0.5
    hold_bars: int = 2


class GeneratedStrategy(BaseStrategy[AccelerationParams]):
    strategy_id = "gen_local_1791228378"

    @classmethod
    def params_type(cls):
        return AccelerationParams

    def warmup_bars(self, params: AccelerationParams) -> int:
        return int(max(params.roc_bars + 3, params.participation_bars + 3))

    def indicators(self, data: pd.DataFrame, params: AccelerationParams) -> pd.DataFrame:
        if params.roc_bars < 1 or params.participation_bars < 3:
            raise ValueError("Lookbacks must be positive; participation_bars must be at least three.")
        if params.hold_bars not in (1, 2):
            raise ValueError("hold_bars must be one or two.")
        if not np.isfinite(params.acceleration_threshold) or params.acceleration_threshold <= 0:
            raise ValueError("acceleration_threshold must be finite and positive.")
        out = pd.DataFrame(index=data.index)
        close = data["close"].where(data["close"] > 0)
        out["return"] = close.pct_change(fill_method=None)
        out["roc"] = close.pct_change(params.roc_bars, fill_method=None)
        out["acceleration"] = out["roc"].diff()
        out["volatility"] = out["return"].rolling(params.participation_bars).std(ddof=0).shift(1)
        out["up_session"] = out["return"].gt(0).astype(float).where(out["return"].notna())
        out["participation"] = out["up_session"].rolling(params.participation_bars).mean()
        out["susceptible"] = 1.0 - out["participation"].shift(1)
        denominator = out["volatility"].where(out["volatility"] > 0) * out["susceptible"].clip(lower=0.2)
        out["propagation"] = (out["acceleration"] / denominator).replace([np.inf, -np.inf], np.nan)
        out["eligible"] = (
            out["roc"].gt(0)
            & out["participation"].between(0.5, 0.8)
            & out["propagation"].gt(params.acceleration_threshold)
        ).fillna(False)
        out["confirmed"] = out["eligible"] & out["eligible"].shift(1, fill_value=False)
        return out

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: AccelerationParams,
    ) -> SignalFrame:
        if params.hold_bars not in (1, 2):
            raise ValueError("hold_bars must be one or two.")
        entries = indicators["confirmed"].fillna(False).to_numpy(dtype=bool)
        signal = np.zeros(len(data), dtype=int)
        exit_at = -1
        for i, enter in enumerate(entries):
            if exit_at >= 0:
                if i >= exit_at:
                    exit_at = -1
                else:
                    signal[i] = 1
            elif enter:
                signal[i] = 1
                exit_at = i + params.hold_bars
        df = pd.DataFrame(index=data.index)
        df["signal"] = signal
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = 0.95
        return SignalFrame(data=df, signal_column="signal", size_column="size")


if __name__ == "__main__":
    strategy = GeneratedStrategy()
    index = pd.date_range("2020-01-01", periods=8)
    data = pd.DataFrame(100.0, index=index, columns=["open", "high", "low", "close", "volume"])
    indicators = pd.DataFrame({"confirmed": [False, True, False, False, False, False, False, False]}, index=index)
    for hold in (1, 2):
        params = AccelerationParams(hold_bars=hold)
        ctx = StrategyContext(symbol="SPY", timeframe="1d", warmup_bars=strategy.warmup_bars(params))
        result = strategy.generate_signals(data, indicators, ctx, params).data
        expected = np.zeros(8, dtype=int)
        expected[2:2 + hold] = 1
        assert np.array_equal(result["signal"].to_numpy(), expected)
        assert result["size"].gt(0).all()
        flat = strategy.indicators(data, params)
        assert not flat["confirmed"].any()
