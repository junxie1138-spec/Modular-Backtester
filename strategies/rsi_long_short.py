from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class RSILongShortParams:
    period: int = 14
    oversold: float = 30.0
    overbought: float = 70.0
    size: float = 1.0


class RSILongShortStrategy(BaseStrategy[RSILongShortParams]):
    """
    Purpose:
        Symmetric RSI mean-reversion. Emit +1 (long) when RSI falls below
        `oversold`, -1 (short) when RSI rises above `overbought`. Hold the
        position until the opposite trigger fires.

    Inputs:
        OHLCV dataframe with datetime index and `close` column.

    Outputs:
        SignalFrame with `signal` in {-1, 0, 1} and `size` columns.

    Side effects:
        None.

    Requires:
        ExecutionConfig.allow_short = True at the config layer. Otherwise the
        portfolio simulator will raise ShortNotAllowedError on the first -1.
    """

    strategy_id = "rsi_long_short"
    version = "1.0"
    asset_type = "stock"
    timeframe = "1d"

    @classmethod
    def params_type(cls):
        return RSILongShortParams

    def warmup_bars(self, params: RSILongShortParams) -> int:
        return params.period + 1

    def indicators(self, data: pd.DataFrame, params: RSILongShortParams) -> pd.DataFrame:
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
        params: RSILongShortParams,
    ) -> SignalFrame:
        rsi = indicators["rsi"]
        # Trigger state: +1 if RSI < oversold, -1 if RSI > overbought, 0 else
        trig = pd.Series(0, index=data.index, dtype="int64")
        trig[rsi < params.oversold] = 1
        trig[rsi > params.overbought] = -1
        # Hold the last non-zero trigger until the opposite fires
        held = trig.replace(0, np.nan).ffill().fillna(0).astype(int)
        df = pd.DataFrame(index=data.index)
        df["signal"] = held.shift(1).fillna(0).astype(int)
        df["size"] = params.size
        return SignalFrame(data=df, signal_column="signal", size_column="size")
