from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class ShockParams:
    lookback: int = 20
    shock_multiple: float = 2.5
    quiet_ratio: float = 0.9
    profit_target: float = 0.025
    max_hold: int = 8


class GeneratedStrategy(BaseStrategy[ShockParams]):
    strategy_id = "gen_local_1791226039"

    @classmethod
    def params_type(cls) -> type[ShockParams]:
        return ShockParams

    @classmethod
    def warmup_bars(cls, params: ShockParams) -> int:
        return max(200, params.lookback + 2, 7)

    def indicators(self, data: pd.DataFrame, params: ShockParams) -> pd.DataFrame:
        ret = data["close"].pct_change().fillna(0.0)
        ma = data["close"].rolling(200, min_periods=200).mean()
        baseline = ret.abs().rolling(params.lookback, min_periods=params.lookback).median().shift(1)
        quiet = ret.abs().rolling(5, min_periods=5).mean().shift(1)
        out = pd.DataFrame(index=data.index)
        out["return"] = ret
        out["ma200"] = ma.fillna(0.0)
        out["baseline"] = baseline.fillna(0.0)
        out["quiet"] = quiet.fillna(0.0)
        out["entry"] = (
            data["close"].gt(ma)
            & baseline.gt(0.0)
            & quiet.le(params.quiet_ratio * baseline)
            & ret.le(-params.shock_multiple * baseline)
        ).fillna(False)
        return out

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: ShockParams,
    ) -> SignalFrame:
        decision = pd.Series(0, index=data.index, dtype=int)
        opens = data["open"].to_numpy()
        closes = data["close"].to_numpy()
        entries = indicators["entry"].to_numpy(dtype=bool)
        in_position = False
        entry_price = 0.0
        entry_bar = -1

        for i in range(len(data)):
            if i > 0:
                if decision.iat[i - 1] == 1 and not in_position:
                    in_position = True
                    entry_price = float(opens[i])
                    entry_bar = i
                elif decision.iat[i - 1] == 0 and in_position:
                    in_position = False

            if in_position:
                if closes[i] >= entry_price * (1.0 + params.profit_target) or i - entry_bar + 1 >= params.max_hold:
                    decision.iat[i] = 0
                else:
                    decision.iat[i] = 1
            elif entries[i]:
                decision.iat[i] = 1

        df = pd.DataFrame(index=data.index)
        df["signal"] = decision
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = 1.0
        return SignalFrame(data=df, signal_column="signal", size_column="size")
