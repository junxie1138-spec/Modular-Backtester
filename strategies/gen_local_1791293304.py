from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class Params:
    capacity_bars: int = 10
    innovation_threshold: float = 0.12
    overflow_threshold: float = 0.10


class GeneratedStrategy(BaseStrategy[Params]):
    strategy_id = "gen_local_1791293304"

    @classmethod
    def params_type(cls) -> type[Params]:
        return Params

    def warmup_bars(self, params: Params) -> int:
        return max(params.capacity_bars + 1, 22)

    def indicators(self, data: pd.DataFrame, params: Params) -> pd.DataFrame:
        if params.capacity_bars < 2 or not np.isfinite(params.innovation_threshold) or not np.isfinite(params.overflow_threshold) or params.innovation_threshold <= 0 or params.overflow_threshold <= 0:
            raise ValueError("Lookback must be at least two and thresholds must be finite and positive")
        close = data["close"].where(data["close"] > 0)
        returns = np.log(close).diff()
        slope = returns.ewm(span=3, adjust=False, min_periods=3).mean()
        scale = returns.rolling(20, min_periods=20).std(ddof=0).shift(1).replace(0, np.nan)
        innovation = slope.diff() / scale
        capacity = data["volume"].rolling(params.capacity_bars, min_periods=params.capacity_bars).mean().shift(1).replace(0, np.nan)
        location = ((2 * close - data["high"] - data["low"]) / (data["high"] - data["low"]).replace(0, np.nan)).clip(-1, 1)
        overflow = location * (data["volume"] / capacity - 1).clip(lower=0)
        result = pd.DataFrame({"returns": returns, "slope": slope, "scale": scale, "innovation": innovation, "capacity": capacity, "location": location, "overflow": overflow}, index=data.index)
        result = result.replace([np.inf, -np.inf], np.nan)
        result["valid"] = result.notna().all(axis=1)
        return result.fillna(0)

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: Params) -> SignalFrame:
        long_entry = indicators["valid"] & (indicators["slope"] > 0) & (indicators["innovation"] > params.innovation_threshold) & (indicators["overflow"] > params.overflow_threshold)
        short_entry = indicators["valid"] & (indicators["slope"] < 0) & (indicators["innovation"] < -params.innovation_threshold) & (indicators["overflow"] < -params.overflow_threshold)
        desired = np.select([long_entry, short_entry], [1, -1], default=0)
        position = 0
        states = np.zeros(len(data), dtype=int)
        for i, entry in enumerate(desired):
            if position == 0:
                position = int(entry)
            elif entry == -position:
                position = 0
            states[i] = position
        df = pd.DataFrame({"signal": states, "size": 0.95}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


if __name__ == "__main__":
    params = Params()
    data = pd.DataFrame(index=pd.date_range("2020-01-01", periods=7))
    evidence = pd.Series([1, 0, -1, -1, 0, 1, 1], index=data.index)
    indicators = pd.DataFrame({"valid": True, "slope": evidence * 0.01, "innovation": evidence, "overflow": evidence}, index=data.index)
    result = GeneratedStrategy.generate_signals(None, data, indicators, None, params).data
    assert result["signal"].tolist() == [0, 1, 1, 0, -1, -1, 0]
    assert result["size"].gt(0).all()
