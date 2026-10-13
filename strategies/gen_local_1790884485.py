from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class VolatilityBalanceParams:
    volatility_window: int = 8
    percentile_window: int = 126
    tail_percentile: float = 0.80


class GeneratedStrategy(BaseStrategy[VolatilityBalanceParams]):
    strategy_id = "gen_local_1790884485"

    @classmethod
    def params_type(cls) -> type[VolatilityBalanceParams]:
        return VolatilityBalanceParams

    def warmup_bars(self, params: VolatilityBalanceParams) -> int:
        return int(params.volatility_window + params.percentile_window + 1)

    def indicators(self, data: pd.DataFrame, params: VolatilityBalanceParams) -> pd.DataFrame:
        if params.volatility_window < 2 or params.percentile_window < 2:
            raise ValueError("Both windows must be at least two bars")
        if not 0.5 < params.tail_percentile < 1.0:
            raise ValueError("tail_percentile must be between 0.5 and 1.0")

        out = pd.DataFrame(index=data.index)
        out["return"] = data["close"].pct_change(fill_method=None).replace([np.inf, -np.inf], np.nan)
        out["down_return"] = out["return"].clip(upper=0.0)
        out["up_return"] = out["return"].clip(lower=0.0)
        out["down_volatility"] = out["down_return"].rolling(params.volatility_window, min_periods=params.volatility_window).std(ddof=0)
        out["up_volatility"] = out["up_return"].rolling(params.volatility_window, min_periods=params.volatility_window).std(ddof=0)
        out["total_directional_volatility"] = out["down_volatility"] + out["up_volatility"]
        out["balance"] = (out["down_volatility"] - out["up_volatility"]) / out["total_directional_volatility"].replace(0.0, np.nan)
        out["upper_threshold"] = out["balance"].rolling(params.percentile_window, min_periods=params.percentile_window).quantile(params.tail_percentile).shift(1)
        out["lower_threshold"] = out["balance"].rolling(params.percentile_window, min_periods=params.percentile_window).quantile(1.0 - params.tail_percentile).shift(1)
        return out

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: VolatilityBalanceParams) -> SignalFrame:
        valid = indicators[["balance", "upper_threshold", "lower_threshold"]].notna().all(axis=1)
        entry = valid & indicators["balance"].gt(0.0) & indicators["balance"].gt(indicators["upper_threshold"])
        reversal = valid & indicators["balance"].lt(0.0) & indicators["balance"].lt(indicators["lower_threshold"])

        decisions = pd.Series(np.nan, index=data.index, dtype=float)
        decisions.loc[entry] = 1.0
        decisions.loc[reversal] = 0.0
        df = pd.DataFrame(index=data.index)
        df["signal"] = decisions.ffill().fillna(0).astype(int)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = 1.0
        return SignalFrame(data=df, signal_column="signal", size_column="size")
