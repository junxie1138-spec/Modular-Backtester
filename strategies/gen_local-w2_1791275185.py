from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class StreakParams:
    tension_bars: int = 2
    release_bars: int = 2


class GeneratedStrategy(BaseStrategy[StreakParams]):
    strategy_id = "gen_local-w2_1791275185"

    @classmethod
    def params_type(cls) -> type[StreakParams]:
        return StreakParams

    @staticmethod
    def warmup_bars(params: StreakParams) -> int:
        if not (1 <= params.tension_bars <= 4 and 1 <= params.release_bars <= 4):
            raise ValueError("Streak lengths must be between 1 and 4")
        return int(max(params.tension_bars, params.release_bars) + params.release_bars + 1)

    @staticmethod
    def indicators(data: pd.DataFrame, params: StreakParams) -> pd.DataFrame:
        GeneratedStrategy.warmup_bars(params)
        result = pd.DataFrame(index=data.index)
        result["change"] = data["close"].diff()
        result["up"] = result["change"].gt(0)
        result["down"] = result["change"].lt(0)
        result["up_streak"] = 0
        result["down_streak"] = 0
        cap = max(params.tension_bars, params.release_bars)
        for length in range(1, cap + 1):
            result["up_streak"] += result["up"].rolling(length, min_periods=length).sum().eq(length).astype(int)
            result["down_streak"] += result["down"].rolling(length, min_periods=length).sum().eq(length).astype(int)
        result["prior_down_streak"] = result["down_streak"].shift(params.release_bars).fillna(0)
        result["prior_up_streak"] = result["up_streak"].shift(params.release_bars).fillna(0)
        return result

    @staticmethod
    def generate_signals(
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: StreakParams,
    ) -> SignalFrame:
        bullish = (
            indicators["up_streak"].ge(params.release_bars)
            & indicators["prior_down_streak"].ge(params.tension_bars)
        )
        bearish = (
            indicators["down_streak"].ge(params.release_bars)
            & indicators["prior_up_streak"].ge(params.tension_bars)
        )
        df = pd.DataFrame(index=data.index)
        df["signal"] = pd.Series(
            np.where(bullish, 1.0, np.where(bearish, 0.0, np.nan)),
            index=data.index,
        ).ffill().fillna(0).astype(int)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = 0.95
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check() -> None:
    index = pd.date_range("2020-01-01", periods=9)
    close = pd.Series([100, 99, 98, 99, 100, 101, 100, 99, 98], index=index, dtype=float)
    data = pd.DataFrame({"open": close, "high": close + 1, "low": close - 1, "close": close, "volume": 1000.0})
    params = StreakParams()
    derived = GeneratedStrategy.indicators(data, params)
    result = GeneratedStrategy.generate_signals(data, derived, None, params).data
    assert result["signal"].tolist() == [0, 0, 0, 0, 0, 1, 1, 1, 0]
    assert result["size"].gt(0).all()
    assert GeneratedStrategy.warmup_bars(StreakParams(4, 4)) == 9


if __name__ == "__main__":
    _self_check()
