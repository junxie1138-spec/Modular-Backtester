from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class VolatilityPersistenceParams:
    volatility_window: int = 20
    aggregation_bars: int = 5
    momentum_bars: int = 10
    variance_ratio_threshold: float = 1.15


class GeneratedStrategy(BaseStrategy[VolatilityPersistenceParams]):
    strategy_id = "gen_local_1791233228"

    @classmethod
    def params_type(cls) -> type[VolatilityPersistenceParams]:
        return VolatilityPersistenceParams

    def warmup_bars(self, params: VolatilityPersistenceParams) -> int:
        return int(max(200, params.volatility_window + params.aggregation_bars, params.momentum_bars + 1))

    def indicators(self, data: pd.DataFrame, params: VolatilityPersistenceParams) -> pd.DataFrame:
        if params.volatility_window < 2 or params.aggregation_bars < 2 or params.momentum_bars < 1:
            raise ValueError("Lookbacks must support volatility and momentum calculations")
        if not np.isfinite(params.variance_ratio_threshold) or params.variance_ratio_threshold <= 0:
            raise ValueError("variance_ratio_threshold must be positive and finite")
        result = pd.DataFrame(index=data.index)
        close = data["close"].astype(float).where(lambda values: (values > 0) & np.isfinite(values))
        result["log_close"] = np.log(close)
        result["daily_return"] = result["log_close"].diff()
        result["aggregate_return"] = result["daily_return"].rolling(params.aggregation_bars, min_periods=params.aggregation_bars).sum()
        result["daily_volatility"] = result["daily_return"].rolling(params.volatility_window, min_periods=params.volatility_window).std(ddof=1)
        result["aggregate_volatility"] = result["aggregate_return"].rolling(params.volatility_window, min_periods=params.volatility_window).std(ddof=1)
        daily_variance = result["daily_volatility"].pow(2).replace(0.0, np.nan)
        result["variance_ratio"] = (result["aggregate_volatility"].pow(2) / (params.aggregation_bars * daily_variance)).replace([np.inf, -np.inf], np.nan)
        result["momentum"] = result["log_close"].diff(params.momentum_bars)
        result["regime_ma"] = close.rolling(200, min_periods=200).mean()
        return result

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: VolatilityPersistenceParams) -> SignalFrame:
        entry_condition = (
            indicators["variance_ratio"].gt(params.variance_ratio_threshold)
            & indicators["momentum"].gt(0.0)
            & data["close"].gt(indicators["regime_ma"])
        ).fillna(False)
        df = pd.DataFrame(index=data.index)
        # The position remains long while the entry condition holds; its reversal exits.
        df["signal"] = entry_condition.astype(int)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = 1.0
        return SignalFrame(data=df, signal_column="signal", size_column="size")
