from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class CompressionParams:
    range_window: int = 8
    reversal_threshold: float = 1.5
    compression_power: float = 1.0


class GeneratedStrategy(BaseStrategy[CompressionParams]):
    strategy_id = "gen_local_1791238129"

    @classmethod
    def params_type(cls) -> type[CompressionParams]:
        return CompressionParams

    def warmup_bars(self, params: CompressionParams) -> int:
        if not 2 <= params.range_window <= 10:
            raise ValueError("range_window must be between 2 and 10")
        if not np.isfinite(params.reversal_threshold) or params.reversal_threshold <= 0:
            raise ValueError("reversal_threshold must be finite and positive")
        if not np.isfinite(params.compression_power) or params.compression_power <= 0:
            raise ValueError("compression_power must be finite and positive")
        return int(params.range_window)

    def indicators(self, data: pd.DataFrame, params: CompressionParams) -> pd.DataFrame:
        self.warmup_bars(params)
        out = pd.DataFrame(index=data.index)
        out["daily_range"] = (data["high"] - data["low"]).where(lambda s: s > 0)
        out["mean_range"] = out["daily_range"].rolling(
            params.range_window, min_periods=params.range_window
        ).mean()
        out["compression"] = (
            1.0 - out["daily_range"] / out["mean_range"]
        ).clip(lower=0.0, upper=1.0)
        out["closing_pressure"] = (
            (2.0 * data["close"] - data["high"] - data["low"])
            / out["daily_range"]
        ).clip(lower=-1.0, upper=1.0)
        out["vote"] = (
            out["compression"].pow(params.compression_power)
            * out["closing_pressure"]
        ).replace([np.inf, -np.inf], np.nan).fillna(0.0)
        return out

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: CompressionParams,
    ) -> SignalFrame:
        self.warmup_bars(params)
        threshold = params.reversal_threshold
        conviction = 0.0
        bullish = False
        decisions = np.zeros(len(data), dtype=int)
        votes = indicators["vote"].replace([np.inf, -np.inf], np.nan).fillna(0.0)
        for i, vote in enumerate(votes.to_numpy(dtype=float)):
            conviction = float(np.clip(conviction + vote, -threshold, threshold))
            if conviction >= threshold:
                bullish = True
            elif conviction <= -threshold:
                bullish = False
            decisions[i] = int(bullish)
        df = pd.DataFrame(index=data.index)
        df["signal"] = decisions
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = 1.0
        return SignalFrame(data=df, signal_column="signal", size_column="size")


if __name__ == "__main__":
    strategy = GeneratedStrategy()
    params = CompressionParams(reversal_threshold=1.0)
    index = pd.date_range("2020-01-01", periods=8)
    data = pd.DataFrame(
        {"open": 100.0, "high": 101.0, "low": 99.0, "close": 100.0, "volume": 1000.0},
        index=index,
    )
    derived = strategy.indicators(data, params)
    assert derived.index.equals(data.index)
    assert np.isfinite(derived["vote"]).all()
    assert strategy.warmup_bars(params) <= 10
    votes = pd.DataFrame({"vote": [0.5, 0.5, 0.0, -0.5, -0.5, -0.5, -0.5, 0.0]}, index=index)
    result = strategy.generate_signals(data, votes, None, params).data
    assert result["signal"].tolist() == [0, 0, 1, 1, 1, 1, 1, 0]
    assert (result["size"] > 0).all()
