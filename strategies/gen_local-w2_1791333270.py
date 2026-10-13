from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class SeasonalStreakParams:
    min_streak: int = 2
    start_weekday: int = 1

    def __post_init__(self):
        if self.min_streak < 1 or not 0 <= self.start_weekday <= 3:
            raise ValueError("min_streak must be positive; start_weekday must be 0..3")


class GeneratedStrategy(BaseStrategy[SeasonalStreakParams]):
    strategy_id = "gen_local-w2_1791333270"

    @classmethod
    def params_type(cls) -> type[SeasonalStreakParams]:
        return SeasonalStreakParams

    def warmup_bars(self, params: SeasonalStreakParams) -> int:
        return max(200, params.min_streak)

    def indicators(self, data: pd.DataFrame, params: SeasonalStreakParams) -> pd.DataFrame:
        out = pd.DataFrame(index=data.index)
        close = data["close"]
        valid = np.isfinite(close) & np.isfinite(data["open"]) & close.gt(0) & data["open"].gt(0)
        out["down_day"] = valid & close.lt(data["open"])
        out["down_streak"] = out["down_day"].astype(int).groupby((~out["down_day"]).cumsum()).cumsum()
        out["ma200"] = close.where(np.isfinite(close) & close.gt(0)).rolling(200, min_periods=200).mean()
        out["bull_regime"] = valid & out["ma200"].notna() & close.gt(out["ma200"])
        weekday = pd.Series(data.index.dayofweek, index=data.index)
        out["seasonal_window"] = weekday.between(params.start_weekday, params.start_weekday + 1)
        return out

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame,
                         ctx: StrategyContext, params: SeasonalStreakParams) -> SignalFrame:
        # A position stays open exactly while the complete entry predicate is true.
        # Leaving the two-weekday seasonal window also reverses that predicate.
        condition = (indicators["bull_regime"] & indicators["seasonal_window"]
                     & indicators["down_streak"].ge(params.min_streak)).fillna(False)
        df = pd.DataFrame(index=data.index)
        df["signal"] = condition.astype(int)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = 0.95
        return SignalFrame(data=df, signal_column="signal", size_column="size")


if __name__ == "__main__":
    index = pd.bdate_range("2020-01-01", periods=220)
    close = pd.Series(100.0 + np.arange(len(index)) * 0.1, index=index)
    data = pd.DataFrame({"open": close - 0.1, "high": close + 2.0,
                         "low": close - 2.0, "close": close, "volume": 1000.0})
    i = next(j for j in range(201, 216) if index[j].dayofweek == 1)
    data.iloc[i - 1:i + 2, data.columns.get_loc("open")] = close.iloc[i - 1:i + 2] + 1.0
    strategy = GeneratedStrategy()
    params = SeasonalStreakParams()
    signals = strategy.generate_signals(data, strategy.indicators(data, params), None, params).data
    assert signals["signal"].iloc[i:i + 4].tolist() == [0, 1, 1, 0]
    assert signals["signal"].iloc[:200].eq(0).all()
    data.iloc[i + 1, data.columns.get_loc("open")] = close.iloc[i + 1] - 0.1
    signals = strategy.generate_signals(data, strategy.indicators(data, params), None, params).data
    assert signals["signal"].iloc[i:i + 4].tolist() == [0, 1, 0, 0]
