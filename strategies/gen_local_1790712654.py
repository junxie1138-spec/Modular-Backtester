from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class GeneratedParams:
    autocorr_window: int = 42
    autocorr_threshold: float = 0.08
    momentum_bars: int = 5
    profit_target_pct: float = 0.06
    max_hold_bars: int = 8


class GeneratedStrategy(BaseStrategy[GeneratedParams]):
    strategy_id = "gen_local_1790712654"

    @classmethod
    def params_type(cls):
        return GeneratedParams

    @classmethod
    def warmup_bars(cls, params: GeneratedParams) -> int:
        return max(200, params.autocorr_window + 1, params.momentum_bars + 1)

    @classmethod
    def indicators(cls, data: pd.DataFrame, params: GeneratedParams) -> pd.DataFrame:
        close = data["close"]
        returns = close.pct_change()
        out = pd.DataFrame(index=data.index)
        out["ma200"] = close.rolling(200, min_periods=200).mean()
        out["autocorr"] = returns.rolling(
            params.autocorr_window, min_periods=params.autocorr_window
        ).corr(returns.shift(1))
        out["recent_return"] = close.pct_change(params.momentum_bars)
        return out

    @classmethod
    def generate_signals(
        cls,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: GeneratedParams,
    ) -> SignalFrame:
        entry = (
            (data["close"] > indicators["ma200"])
            & (indicators["autocorr"] >= params.autocorr_threshold)
            & (indicators["recent_return"] > 0)
        ).fillna(False).to_numpy(dtype=bool)
        close = data["close"].to_numpy(dtype=float)
        raw = np.zeros(len(data), dtype=int)
        in_position = False
        entry_price = 0.0
        held_bars = 0
        exited = False

        for i in range(len(data)):
            exited = False
            if in_position:
                held_bars += 1
                if (
                    close[i] >= entry_price * (1.0 + params.profit_target_pct)
                    or held_bars >= params.max_hold_bars
                ):
                    in_position = False
                    exited = True
            if not in_position and not exited and entry[i]:
                in_position = True
                entry_price = close[i]
                held_bars = 0
                raw[i] = 1
            elif in_position:
                raw[i] = 1

        df = data.copy()
        df["signal"] = pd.Series(raw, index=data.index).shift(1).fillna(0).astype(int)
        df["size"] = 1.0
        return SignalFrame(data=df, signal_column="signal", size_column="size")
