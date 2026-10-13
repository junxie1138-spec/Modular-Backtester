from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class FractionalBreakoutParams:
    fractional_order: float = 0.45
    channel_bars: int = 20


class GeneratedStrategy(BaseStrategy[FractionalBreakoutParams]):
    strategy_id = "gen_local_1790985066"

    @classmethod
    def params_type(cls) -> type[FractionalBreakoutParams]:
        return FractionalBreakoutParams

    def warmup_bars(self, params: FractionalBreakoutParams) -> int:
        return 64 + 5 + max(2, int(params.channel_bars))

    def indicators(self, data: pd.DataFrame, params: FractionalBreakoutParams) -> pd.DataFrame:
        result = pd.DataFrame(index=data.index)
        close = data["close"].astype(float).replace([np.inf, -np.inf], np.nan)
        order = float(params.fractional_order)
        channel = max(2, int(params.channel_bars))

        # Fixed 64-term causal approximation of (1 - lag)^order.
        fractional_close = close.copy()
        weight = 1.0
        for lag in range(1, 64):
            weight *= (lag - 1 - order) / lag
            fractional_close = fractional_close + weight * close.shift(lag)

        result["fractional_close"] = fractional_close
        result["wave"] = result["fractional_close"].rolling(5, min_periods=5).mean()
        prior_wave = result["wave"].shift(1)
        result["upper_channel"] = prior_wave.rolling(channel, min_periods=channel).max()
        result["lower_channel"] = prior_wave.rolling(channel, min_periods=channel).min()
        result["valid"] = result[["wave", "upper_channel", "lower_channel"]].notna().all(axis=1)
        result["bullish_breakout"] = result["valid"] & (result["wave"] > result["upper_channel"])
        result["bearish_breakout"] = result["valid"] & (result["wave"] < result["lower_channel"])
        return result

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: FractionalBreakoutParams,
    ) -> SignalFrame:
        df = pd.DataFrame(index=data.index)
        # The latest directional escape defines the persistent entry condition.
        # Only an opposite escape flips that condition and exits the position.
        regime_change = pd.Series(np.nan, index=data.index, dtype=float)
        regime_change.loc[indicators["bullish_breakout"]] = 1.0
        regime_change.loc[indicators["bearish_breakout"]] = 0.0
        df["signal"] = regime_change.ffill().fillna(0).astype(int)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = 0.95
        return SignalFrame(data=df, signal_column="signal", size_column="size")
