from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class StrategyParams:
    streak_days: int = 4
    gap_pct: float = 0.002
    profit_pct: float = 0.025
    max_hold: int = 8
    size: float = 1.0


class GeneratedStrategy(BaseStrategy[StrategyParams]):
    strategy_id = "gen_local-w2_1791296252"

    @classmethod
    def params_type(cls):
        return StrategyParams

    def warmup_bars(self, params: StrategyParams) -> int:
        return 200

    def indicators(self, data: pd.DataFrame, params: StrategyParams) -> pd.DataFrame:
        close = data["close"]
        direction = np.sign(close.diff().fillna(0)).astype(int)
        runs = direction.ne(direction.shift()).cumsum()
        out = pd.DataFrame(index=data.index)
        out["streak"] = direction * (direction.groupby(runs).cumcount() + 1)
        out["gap"] = (data["open"].div(close.shift(1)).sub(1)
                      .replace([np.inf, -np.inf], np.nan).fillna(0))
        out["ma200"] = close.rolling(200, min_periods=200).mean().fillna(0)
        return out

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: StrategyParams,
    ) -> SignalFrame:
        close = data["close"].to_numpy()
        opens = data["open"].to_numpy()
        streak = indicators["streak"].to_numpy()
        gap = indicators["gap"].to_numpy()
        ma = indicators["ma200"].to_numpy()
        decision = np.zeros(len(data), dtype=int)
        entry_price = 0.0
        entry_bar = 0
        previous_position = 0

        for i in range(len(data)):
            position = int(decision[i - 1]) if i else 0
            if position:
                if position != previous_position:
                    entry_price = float(opens[i])
                    entry_bar = i
                gain = position * (close[i] / entry_price - 1)
                if gain < params.profit_pct and i - entry_bar + 1 < params.max_hold:
                    decision[i] = position
            elif ma[i] > 0:
                if (streak[i] == params.streak_days and
                        gap[i] <= -params.gap_pct and close[i] > ma[i]):
                    decision[i] = 1
                elif (streak[i] == -params.streak_days and
                      gap[i] >= params.gap_pct and close[i] < ma[i]):
                    decision[i] = -1
            previous_position = position

        df = pd.DataFrame(index=data.index)
        df["signal"] = decision
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = float(params.size)
        return SignalFrame(data=df, signal_column="signal", size_column="size")
