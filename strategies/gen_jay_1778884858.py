from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


def _consec_streak(cond: pd.Series) -> pd.Series:
    flag = cond.astype(bool)
    cumulative = flag.cumsum()
    last_false = cumulative.where(~flag).ffill().fillna(0)
    return (cumulative - last_false) * flag.astype(int)


@dataclass(slots=True)
class Params:
    price_streak_min: int = 2
    volume_streak_min: int = 2
    atr_period: int = 14
    atr_multiplier: float = 2.0


class GeneratedStrategy(BaseStrategy[Params]):
    strategy_id = "gen_jay_1778884858"

    @classmethod
    def params_type(cls):
        return Params

    @staticmethod
    def warmup_bars(params: Params) -> int:
        return params.atr_period + 2

    @staticmethod
    def indicators(data: pd.DataFrame, params: Params) -> pd.DataFrame:
        ind = pd.DataFrame(index=data.index)

        close = data["close"]
        volume = data["volume"]
        high = data["high"]
        low = data["low"]

        ind["price_streak"] = _consec_streak(close > close.shift(1))
        ind["volume_streak"] = _consec_streak(volume > volume.shift(1))

        prev_close = close.shift(1)
        tr = pd.concat(
            [
                high - low,
                (high - prev_close).abs(),
                (low - prev_close).abs(),
            ],
            axis=1,
        ).max(axis=1)
        ind["atr"] = tr.rolling(params.atr_period).mean()

        return ind

    @staticmethod
    def generate_signals(
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: Params,
    ) -> SignalFrame:
        close = data["close"].to_numpy()
        price_streak = indicators["price_streak"].to_numpy()
        volume_streak = indicators["volume_streak"].to_numpy()
        atr = indicators["atr"].to_numpy()

        n = len(close)
        signal = np.zeros(n, dtype=int)
        size = np.ones(n, dtype=float)

        in_position = False
        stop_level = 0.0

        for i in range(n):
            if np.isnan(atr[i]):
                continue

            if in_position:
                if close[i] < stop_level:
                    signal[i] = 0
                    in_position = False
                else:
                    signal[i] = 1
            else:
                if (
                    price_streak[i] >= params.price_streak_min
                    and volume_streak[i] >= params.volume_streak_min
                ):
                    signal[i] = 1
                    in_position = True
                    stop_level = close[i] - params.atr_multiplier * atr[i]

        df = pd.DataFrame({"signal": signal, "size": size}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)

        return SignalFrame(data=df, signal_column="signal", size_column="size")
