from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class TrendInnovationParams:
    lookback: int = 16
    holding_bars: int = 4


class GeneratedStrategy(BaseStrategy[TrendInnovationParams]):
    strategy_id = "gen_local_1791218454"

    @classmethod
    def params_type(cls) -> type[TrendInnovationParams]:
        return TrendInnovationParams

    def warmup_bars(self, params: TrendInnovationParams) -> int:
        return int(params.lookback) + 1

    def indicators(self, data: pd.DataFrame, params: TrendInnovationParams) -> pd.DataFrame:
        length = int(params.lookback)
        if length < 3:
            raise ValueError("lookback must be at least 3")
        out = pd.DataFrame(index=data.index)
        out["log_close"] = np.log(data["close"].where(data["close"] > 0))
        out["log_return"] = out["log_close"].diff()
        out["mean"] = out["log_close"].rolling(length, min_periods=length).mean()
        weights = np.arange(length, dtype=float) - (length - 1) / 2.0
        denominator = float(np.dot(weights, weights))
        slope = np.full(len(data), np.nan, dtype=float)
        if len(data) >= length:
            slope[length - 1:] = np.convolve(
                out["log_close"].to_numpy(dtype=float), weights[::-1], mode="valid"
            ) / denominator
        out["slope"] = slope
        out["slope_innovation"] = out["slope"].diff()
        out["forecast"] = out["mean"].shift(1) + out["slope"].shift(1) * ((length + 1) / 2.0)
        out["volatility"] = out["log_return"].rolling(length, min_periods=length).std(ddof=0)
        scale = out["volatility"].where(out["volatility"] > 0)
        out["forecast_surprise"] = (out["log_close"] - out["forecast"]) / scale
        out["channel_high"] = data["high"].rolling(length, min_periods=length).max().shift(1)
        required = ["slope", "slope_innovation", "forecast_surprise", "channel_high"]
        valid = pd.Series(np.isfinite(out[required].to_numpy()).all(axis=1), index=data.index)
        out["entry"] = (
            valid
            & (data["close"] > out["channel_high"])
            & (out["slope"] > 0)
            & (out["slope_innovation"] > 0)
            & (out["forecast_surprise"] > 1.0)
        ).fillna(False)
        return out

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: TrendInnovationParams,
    ) -> SignalFrame:
        holding = int(params.holding_bars)
        if holding < 3 or holding > 5:
            raise ValueError("holding_bars must be between 3 and 5")
        entries = indicators["entry"].fillna(False).to_numpy(dtype=bool)
        signals = np.zeros(len(data), dtype=int)
        entry_bar = -1
        for i in range(len(data)):
            if entry_bar >= 0:
                if i - entry_bar == holding:
                    entry_bar = -1
                else:
                    signals[i] = 1
            elif entries[i]:
                entry_bar = i
                signals[i] = 1
        df = pd.DataFrame(index=data.index)
        df["signal"] = signals
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = 1.0
        return SignalFrame(data=df, signal_column="signal", size_column="size")


if __name__ == "__main__":
    index = pd.date_range("2020-01-01", periods=12, freq="D")
    data = pd.DataFrame(index=index)
    indicators = pd.DataFrame({"entry": True}, index=index)
    strategy = GeneratedStrategy()
    result = strategy.generate_signals(data, indicators, None, TrendInnovationParams())
    assert result.data["signal"].tolist() == [0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1]
    assert (result.data["size"] > 0).all()
