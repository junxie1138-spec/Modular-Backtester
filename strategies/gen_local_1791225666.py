from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class OrdinalFlipParams:
    entropy_floor: float = 0.70
    profit_target: float = 0.025
    max_hold: int = 8


class GeneratedStrategy(BaseStrategy[OrdinalFlipParams]):
    strategy_id = "gen_local_1791225666"

    @classmethod
    def params_type(cls):
        return OrdinalFlipParams

    def warmup_bars(self, params: OrdinalFlipParams) -> int:
        return 9

    def indicators(self, data: pd.DataFrame, params: OrdinalFlipParams) -> pd.DataFrame:
        up = data["close"].gt(data["close"].shift(1))
        pattern = 2 * up.astype(int) + up.shift(1)
        pattern.iloc[:2] = np.nan
        frequencies = pd.concat(
            [pattern.eq(k).where(pattern.notna()).rolling(6, min_periods=6).mean() for k in range(4)],
            axis=1,
        )
        entropy = -(frequencies * np.log(frequencies.where(frequencies > 0, 1))).sum(
            axis=1, min_count=4
        ) / np.log(4)
        out = pd.DataFrame(index=data.index)
        out["complexity"] = entropy.fillna(0.0)
        out["flip_rate"] = (
            up.ne(up.shift(1)).where(pattern.notna()).rolling(5, min_periods=5).mean().fillna(0.0)
        )
        return out

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: OrdinalFlipParams,
    ) -> SignalFrame:
        complexity = indicators["complexity"]
        entry = (
            complexity.ge(params.entropy_floor)
            & complexity.gt(complexity.shift(1))
            & indicators["flip_rate"].ge(0.6)
            & data["close"].lt(data["close"].shift(1))
        ).fillna(False).to_numpy(dtype=bool)
        opens = data["open"].to_numpy(dtype=float)
        closes = data["close"].to_numpy(dtype=float)
        raw = np.zeros(len(data), dtype=int)
        held = False
        pending_entry = False
        entry_price = 0.0
        entry_bar = 0
        for i in range(len(data)):
            if pending_entry:
                entry_price = opens[i]
                entry_bar = i
                pending_entry = False
            if held:
                if closes[i] >= entry_price * (1 + params.profit_target) or i - entry_bar + 1 >= params.max_hold:
                    held = False
            elif entry[i] and i + 1 < len(data):
                held = True
                pending_entry = True
            raw[i] = int(held)
        df = pd.DataFrame(index=data.index)
        df["signal"] = raw
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = 1.0
        return SignalFrame(data=df, signal_column="signal", size_column="size")
