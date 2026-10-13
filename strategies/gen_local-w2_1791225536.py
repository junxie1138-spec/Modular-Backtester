from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class StreakParams:
    min_streak: int = 3
    close_location: float = 0.6
    stop_atr: float = 1.5


class GeneratedStrategy(BaseStrategy[StreakParams]):
    strategy_id = "gen_local-w2_1791225536"

    @classmethod
    def params_type(cls):
        return StreakParams

    def warmup_bars(self, params: StreakParams) -> int:
        return 200

    def indicators(self, data: pd.DataFrame, params: StreakParams) -> pd.DataFrame:
        close = data["close"]
        down = close.diff().lt(0)
        previous_close = close.shift(1)
        true_range = pd.concat(
            [data["high"] - data["low"],
             (data["high"] - previous_close).abs(),
             (data["low"] - previous_close).abs()],
            axis=1,
        ).max(axis=1)
        out = pd.DataFrame(index=data.index)
        out["streak"] = down.groupby((~down).cumsum()).cumsum().astype(int)
        out["return"] = close.div(previous_close).sub(1).fillna(0.0)
        out["close_location"] = close.sub(data["low"]).div(
            (data["high"] - data["low"]).replace(0, np.nan)
        ).fillna(0.0)
        out["ma200"] = close.rolling(200, min_periods=200).mean().fillna(np.inf)
        out["atr"] = true_range.rolling(14, min_periods=14).mean().fillna(0.0)
        return out

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: StreakParams,
    ) -> SignalFrame:
        close = data["close"]
        setup = (
            (indicators["streak"] >= params.min_streak)
            & (indicators["return"] > indicators["return"].shift(1).fillna(0.0))
            & (indicators["close_location"] >= params.close_location)
            & (close > indicators["ma200"])
            & (indicators["atr"] > 0)
        ).to_numpy()
        closes = close.to_numpy()
        opens = data["open"].to_numpy()
        averages = indicators["ma200"].to_numpy()
        atr = indicators["atr"].to_numpy()
        targets = np.zeros(len(data), dtype=int)
        entry_price = stop_price = 0.0
        age = 0
        for i in range(len(data)):
            if i and targets[i - 1] == 1 and (i == 1 or targets[i - 2] == 0):
                entry_price = float(opens[i])
                stop_price = entry_price - params.stop_atr * float(atr[i - 1])
                age = 0
            if i and targets[i - 1] == 1:
                age += 1
                targets[i] = int(not (
                    closes[i] < stop_price
                    or closes[i] > closes[i - 1]
                    or closes[i] < averages[i]
                    or age >= 2
                ))
            elif setup[i]:
                targets[i] = 1
        df = pd.DataFrame(index=data.index)
        df["signal"] = targets
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = 1.0
        return SignalFrame(data=df, signal_column="signal", size_column="size")


if __name__ == "__main__":
    prices = pd.Series([100.0] * 200 + [120.0, 119.0, 118.5, 118.25, 119.0, 119.1])
    sample = pd.DataFrame({
        "open": prices + 0.1,
        "high": prices + 0.1,
        "low": prices - 1.0,
        "close": prices,
        "volume": 1.0,
    })
    strategy = GeneratedStrategy()
    params = StreakParams()
    result = strategy.generate_signals(sample, strategy.indicators(sample, params), None, params)
    assert result.data["signal"].iloc[-2:].tolist() == [1, 0]
