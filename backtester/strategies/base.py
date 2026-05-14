from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Generic, TypeVar
import pandas as pd

from backtester.core.constants import REQUIRED_OHLCV_COLUMNS
from backtester.core.types import SignalFrame, StrategyContext

P = TypeVar("P")


class BaseStrategy(ABC, Generic[P]):
    strategy_id: str
    version: str = "1.0"
    asset_type: str = "stock"
    timeframe: str = "1d"

    # v0.4.0 opt-in attributes (default False keeps v0.3.0 strategies unchanged):
    uses_multi_symbol: bool = False
    uses_per_bar: bool = False

    @classmethod
    @abstractmethod
    def params_type(cls):
        raise NotImplementedError

    @abstractmethod
    def indicators(self, data: pd.DataFrame, params: P) -> pd.DataFrame:
        raise NotImplementedError

    @abstractmethod
    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: P,
    ) -> SignalFrame:
        raise NotImplementedError

    def validate(self, data: pd.DataFrame, params: P) -> None:
        required = set(REQUIRED_OHLCV_COLUMNS)
        present = set(map(str.lower, data.columns))
        missing = required - present
        if missing:
            raise ValueError(f"Missing required columns: {sorted(missing)}")

    def warmup_bars(self, params: P) -> int:
        return 0
