from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class AccelerationParams:
    roc_bars: int = 5
    smooth_bars: int = 3
    vol_bars: int = 20
    entry_threshold: float = 0.35
    target_annual_vol: float = 0.10
    max_size: float = 0.95


class GeneratedStrategy(BaseStrategy[AccelerationParams]):
    strategy_id = "gen_local_1791330354"

    @classmethod
    def params_type(cls) -> type[AccelerationParams]:
        return AccelerationParams

    def warmup_bars(self, params: AccelerationParams) -> int:
        return int(max(params.vol_bars + 1, 2 * params.roc_bars + params.smooth_bars) + 1)

    def indicators(self, data: pd.DataFrame, params: AccelerationParams) -> pd.DataFrame:
        result = pd.DataFrame(index=data.index)
        close = data["close"].where(data["close"] > 0)
        result["daily_return"] = close.pct_change(fill_method=None)
        result["daily_vol"] = result["daily_return"].rolling(params.vol_bars, min_periods=params.vol_bars).std(ddof=0)
        result["roc"] = close.pct_change(params.roc_bars, fill_method=None)
        result["roc_acceleration"] = result["roc"] - result["roc"].shift(params.roc_bars)
        result["smoothed_acceleration"] = result["roc_acceleration"].rolling(params.smooth_bars, min_periods=params.smooth_bars).mean()
        denominator = result["daily_vol"].replace(0.0, np.nan) * np.sqrt(params.roc_bars)
        result["score"] = result["smoothed_acceleration"] / denominator
        result["risk_size"] = (params.target_annual_vol / (np.sqrt(252.0) * result["daily_vol"].replace(0.0, np.nan))).clip(lower=0.01, upper=params.max_size)
        return result.replace([np.inf, -np.inf], np.nan)

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: AccelerationParams) -> SignalFrame:
        score = indicators["score"]
        valid = score.notna() & indicators["daily_vol"].gt(0)
        candidate = pd.Series(0, index=data.index, dtype=int)
        candidate.loc[valid & score.ge(params.entry_threshold)] = 1
        candidate.loc[valid & score.le(-params.entry_threshold)] = -1
        confirmed = candidate.where(candidate.eq(candidate.shift(1)), 0)
        position = 0
        signals = np.zeros(len(data), dtype=int)
        for i, direction in enumerate(confirmed.to_numpy(dtype=int)):
            if position == 0:
                position = int(direction)
            elif direction == -position:
                position = 0
            signals[i] = position
        df = pd.DataFrame(index=data.index)
        df["signal"] = signals
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = indicators["risk_size"].shift(1).fillna(0.01).astype(float)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


if __name__ == "__main__":
    strategy = GeneratedStrategy()
    params = AccelerationParams()
    index = pd.date_range("2020-01-01", periods=7)
    data = pd.DataFrame({"open": 100.0, "high": 101.0, "low": 99.0, "close": 100.0, "volume": 1000.0}, index=index)
    indicators = pd.DataFrame({"score": [1.0, 1.0, 0.0, -1.0, -1.0, -1.0, -1.0], "daily_vol": 0.01, "risk_size": 0.5}, index=index)
    output = strategy.generate_signals(data, indicators, None, params).data
    assert output["signal"].tolist() == [0, 0, 1, 1, 1, 0, -1]
    assert output["size"].gt(0).all()
