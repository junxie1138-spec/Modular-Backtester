from __future__ import annotations

from dataclasses import dataclass
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class Breakout20DParams:
    lookback: int = 20
    exit_lookback: int = 10
    size: float = 1.0


class Breakout20DStrategy(BaseStrategy[Breakout20DParams]):
    """
    Purpose:
        Long-only Donchian breakout: enter long when close exceeds the rolling
        lookback-day high; exit when close falls below the rolling
        exit_lookback-day low.

    Inputs:
        OHLCV dataframe with datetime index and `close` column.

    Outputs:
        SignalFrame with `signal` (0/1) and `size` columns.

    Side effects:
        None.
    """

    strategy_id = "breakout_20d"
    version = "1.0"
    asset_type = "stock"
    timeframe = "1d"

    @classmethod
    def params_type(cls):
        return Breakout20DParams

    def warmup_bars(self, params: Breakout20DParams) -> int:
        return max(params.lookback, params.exit_lookback)

    def indicators(self, data: pd.DataFrame, params: Breakout20DParams) -> pd.DataFrame:
        out = pd.DataFrame(index=data.index)
        out["upper"] = data["close"].rolling(params.lookback).max().shift(1)
        out["lower"] = data["close"].rolling(params.exit_lookback).min().shift(1)
        return out

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: Breakout20DParams,
    ) -> SignalFrame:
        close = data["close"]
        long_trigger = (close > indicators["upper"]).astype(int)
        exit_trigger = (close < indicators["lower"]).astype(int) * -1
        raw = (long_trigger + exit_trigger).replace(0, pd.NA).ffill().fillna(0).clip(lower=0).astype(int)
        df = pd.DataFrame(index=data.index)
        df["signal"] = raw.shift(1).fillna(0).astype(int)
        df["size"] = params.size
        return SignalFrame(data=df, signal_column="signal", size_column="size")
