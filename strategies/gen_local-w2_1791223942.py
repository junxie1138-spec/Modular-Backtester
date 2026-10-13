from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class StrategyParams:
    channel: int = 8
    streak: int = 2
    size_floor: float = 0.35


class GeneratedStrategy(BaseStrategy[StrategyParams]):
    strategy_id = "gen_local-w2_1791223942"

    @classmethod
    def params_type(cls):
        return StrategyParams

    def warmup_bars(self, params: StrategyParams) -> int:
        return params.channel + 1

    def indicators(self, data: pd.DataFrame, params: StrategyParams) -> pd.DataFrame:
        out = pd.DataFrame(index=data.index)
        higher_low = data["low"].gt(data["low"].shift(1)).fillna(False)
        lower_high = data["high"].lt(data["high"].shift(1)).fillna(False)
        out["up_streak"] = higher_low.astype(int).groupby((~higher_low).cumsum()).cumsum()
        out["down_streak"] = lower_high.astype(int).groupby((~lower_high).cumsum()).cumsum()
        out["upper"] = data["high"].shift(1).rolling(params.channel, min_periods=params.channel).max().fillna(np.inf)
        out["lower"] = data["low"].shift(1).rolling(params.channel, min_periods=params.channel).min().fillna(-np.inf)
        return out

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: StrategyParams,
    ) -> SignalFrame:
        entry = data["close"].gt(indicators["upper"]) & indicators["up_streak"].ge(params.streak)
        exit_signal = data["close"].lt(indicators["lower"]) & indicators["down_streak"].ge(params.streak)
        events = pd.Series(np.nan, index=data.index)
        events.loc[entry] = 1
        events.loc[exit_signal] = 0
        df = pd.DataFrame(index=data.index)
        df["signal"] = events.ffill().fillna(0).astype(int)
        df["size"] = np.clip(params.size_floor + 0.12 * indicators["up_streak"], params.size_floor, 1.0).shift(1).fillna(params.size_floor).astype(float)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


if __name__ == "__main__":
    params = StrategyParams(channel=3, streak=2)
    close = pd.Series([10, 10, 10, 11, 12, 13, 12, 11, 10, 9], dtype=float)
    sample = pd.DataFrame({"open": close, "high": close + 0.1, "low": close - 0.1, "close": close, "volume": 1})
    strategy = GeneratedStrategy()
    result = strategy.generate_signals(sample, strategy.indicators(sample, params), StrategyContext("SPY", "1d", strategy.warmup_bars(params)), params).data
    assert result["signal"].iloc[4] == 0 and result["signal"].iloc[5] == 1
    assert result["signal"].iloc[7] == 1 and result["signal"].iloc[8] == 0
    assert result["size"].gt(0).all()
