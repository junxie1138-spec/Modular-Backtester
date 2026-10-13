from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class GeneratedParams:
    lookback: int = 63
    drawdown_threshold: float = 0.05


class GeneratedStrategy(BaseStrategy[GeneratedParams]):
    strategy_id = "gen_local_1790712782"

    @classmethod
    def params_type(cls):
        return GeneratedParams

    @classmethod
    def warmup_bars(cls, params: GeneratedParams) -> int:
        return params.lookback

    @classmethod
    def indicators(cls, data: pd.DataFrame, params: GeneratedParams) -> pd.DataFrame:
        rolling_peak = data["close"].rolling(params.lookback, min_periods=params.lookback).max()
        drawdown = data["close"] / rolling_peak - 1.0
        higher_close = data["close"].gt(data["close"].shift(1))
        entry_setup = drawdown.le(-params.drawdown_threshold) & higher_close
        confirmed = entry_setup & entry_setup.shift(1, fill_value=False)
        return pd.DataFrame({"drawdown": drawdown, "entry_setup": entry_setup, "confirmed": confirmed}, index=data.index)

    @classmethod
    def generate_signals(
        cls,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: GeneratedParams,
    ) -> SignalFrame:
        df = pd.DataFrame(index=data.index)
        df["signal"] = indicators["confirmed"].fillna(False).astype(int)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = np.ones(len(df), dtype=float)
        return SignalFrame(data=df, signal_column="signal", size_column="size")
