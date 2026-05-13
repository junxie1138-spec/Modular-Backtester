from __future__ import annotations

from dataclasses import dataclass
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class StrategyParams:
    lookback: int = 20
    size: float = 1.0


class StrategyName(BaseStrategy[StrategyParams]):
    """
    Purpose:
        Replace with one-sentence description.

    Inputs:
        OHLCV dataframe with datetime index and lowercase columns:
        open, high, low, close, volume.

    Outputs:
        SignalFrame with `signal` (0/1) and optional `size` columns.

    Side effects:
        None.
    """

    strategy_id = "replace_me"
    version = "1.0"
    asset_type = "stock"
    timeframe = "1d"

    @classmethod
    def params_type(cls):
        return StrategyParams

    def warmup_bars(self, params: StrategyParams) -> int:
        return params.lookback

    def indicators(self, data: pd.DataFrame, params: StrategyParams) -> pd.DataFrame:
        out = pd.DataFrame(index=data.index)
        return out

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: StrategyParams,
    ) -> SignalFrame:
        df = pd.DataFrame(index=data.index)
        df["signal"] = 0
        df["size"] = params.size
        return SignalFrame(data=df, signal_column="signal", size_column="size")
