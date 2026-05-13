from __future__ import annotations

from dataclasses import dataclass
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class SMACrossParams:
    fast: int = 20
    slow: int = 50
    size: float = 1.0


class SMACrossStrategy(BaseStrategy[SMACrossParams]):
    """
    Purpose:
        Trend-following long-only strategy using fast/slow moving average crossover.

    Inputs:
        OHLCV dataframe with datetime index and `close` column.

    Outputs:
        SignalFrame with `signal` (0/1) and `size` columns.

    Side effects:
        None.
    """

    strategy_id = "sma_cross"
    version = "1.0"
    asset_type = "stock"
    timeframe = "1d"

    @classmethod
    def params_type(cls):
        return SMACrossParams

    def warmup_bars(self, params: SMACrossParams) -> int:
        return max(params.fast, params.slow)

    def indicators(self, data: pd.DataFrame, params: SMACrossParams) -> pd.DataFrame:
        out = pd.DataFrame(index=data.index)
        out["fast_sma"] = data["close"].rolling(params.fast).mean()
        out["slow_sma"] = data["close"].rolling(params.slow).mean()
        return out

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: SMACrossParams,
    ) -> SignalFrame:
        df = pd.DataFrame(index=data.index)
        df["signal"] = 0
        df.loc[indicators["fast_sma"] > indicators["slow_sma"], "signal"] = 1
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = params.size
        return SignalFrame(data=df, signal_column="signal", size_column="size")
