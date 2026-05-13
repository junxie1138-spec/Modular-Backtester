from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class RSIMeanReversionParams:
    period: int = 14
    oversold: float = 30.0
    overbought: float = 70.0
    size: float = 1.0


class RSIMeanReversionStrategy(BaseStrategy[RSIMeanReversionParams]):
    """
    Purpose:
        Long-only mean-reversion: enter long when RSI crosses below `oversold`,
        exit when RSI crosses above `overbought`.

    Inputs:
        OHLCV dataframe with datetime index and `close` column.

    Outputs:
        SignalFrame with `signal` (0/1) and `size` columns.

    Side effects:
        None.
    """

    strategy_id = "rsi_mean_reversion"
    version = "1.0"
    asset_type = "stock"
    timeframe = "1d"

    @classmethod
    def params_type(cls):
        return RSIMeanReversionParams

    def warmup_bars(self, params: RSIMeanReversionParams) -> int:
        return params.period + 1

    def indicators(self, data: pd.DataFrame, params: RSIMeanReversionParams) -> pd.DataFrame:
        out = pd.DataFrame(index=data.index)
        delta = data["close"].diff()
        gain = delta.clip(lower=0.0)
        loss = -delta.clip(upper=0.0)
        avg_gain = gain.ewm(alpha=1.0 / params.period, adjust=False, min_periods=params.period).mean()
        avg_loss = loss.ewm(alpha=1.0 / params.period, adjust=False, min_periods=params.period).mean()
        rs = avg_gain / avg_loss.replace(0.0, np.nan)
        out["rsi"] = 100.0 - (100.0 / (1.0 + rs))
        return out

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: RSIMeanReversionParams,
    ) -> SignalFrame:
        rsi = indicators["rsi"]
        df = pd.DataFrame(index=data.index)
        # State machine: long when last cross was below oversold; flat when last cross was above overbought
        state = (rsi < params.oversold).astype(int) - (rsi > params.overbought).astype(int)
        # Forward-fill the binary state so a long position is held until exit
        signal = state.replace(0, np.nan).ffill().fillna(0).clip(lower=0).astype(int)
        df["signal"] = signal.shift(1).fillna(0).astype(int)
        df["size"] = params.size
        return SignalFrame(data=df, signal_column="signal", size_column="size")
