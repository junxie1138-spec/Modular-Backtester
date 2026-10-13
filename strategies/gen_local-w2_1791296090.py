from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class OrdinalParams:
    complexity_window: int = 12
    horizon: int = 3
    complexity_cutoff: float = 0.68


class GeneratedStrategy(BaseStrategy[OrdinalParams]):
    strategy_id = "gen_local-w2_1791296090"

    @classmethod
    def params_type(cls):
        return OrdinalParams

    def warmup_bars(self, params: OrdinalParams) -> int:
        return max(params.complexity_window + 3, params.horizon + 1)

    def indicators(self, data: pd.DataFrame, params: OrdinalParams) -> pd.DataFrame:
        returns = data["close"].pct_change(fill_method=None).replace([np.inf, -np.inf], np.nan)
        a, b, c = returns.shift(2), returns.shift(1), returns
        valid = a.notna() & b.notna() & c.notna()
        pattern = (a > b).astype(int) + 2 * (a > c).astype(int) + 4 * (b > c).astype(int)
        probabilities = pd.concat(
            [pattern.eq(code).where(valid).rolling(params.complexity_window, min_periods=params.complexity_window).mean()
             for code in (0, 1, 3, 4, 6, 7)],
            axis=1,
        )
        entropy = -(probabilities * np.log(probabilities.where(probabilities.gt(0)))).sum(axis=1) / np.log(6)
        ready = valid.rolling(params.complexity_window, min_periods=params.complexity_window).sum().eq(params.complexity_window)
        move = data["close"].pct_change(params.horizon, fill_method=None).replace([np.inf, -np.inf], np.nan).fillna(0.0)
        out = pd.DataFrame(index=data.index)
        out["ordinal_entropy"] = entropy.fillna(0.0).clip(0.0, 1.0)
        out["signed_move"] = pd.Series(np.where(out["ordinal_entropy"] < params.complexity_cutoff, move, -move), index=data.index).where(ready, 0.0)
        return out

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: OrdinalParams,
    ) -> SignalFrame:
        score = indicators["signed_move"]
        df = pd.DataFrame(index=data.index)
        df["signal"] = pd.Series(np.where(score > 0, 1.0, np.where(score < 0, 0.0, np.nan)), index=data.index).ffill().fillna(0).astype(int)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = 1.0
        return SignalFrame(data=df, signal_column="signal", size_column="size")
