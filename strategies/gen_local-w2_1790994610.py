from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class RangeAccumulationParams:
    fast_window: int = 18
    slow_window: int = 63
    pressure_window: int = 18
    contraction_threshold: float = 0.92
    pressure_threshold: float = 0.12
    target_volatility: float = 0.12
    min_size: float = 0.10
    max_size: float = 0.95


class GeneratedStrategy(BaseStrategy[RangeAccumulationParams]):
    strategy_id = "gen_local-w2_1790994610"

    @classmethod
    def params_type(cls) -> type[RangeAccumulationParams]:
        return RangeAccumulationParams

    def warmup_bars(self, params: RangeAccumulationParams) -> int:
        return int(max(params.fast_window, params.slow_window, params.pressure_window))

    def indicators(self, data: pd.DataFrame, params: RangeAccumulationParams) -> pd.DataFrame:
        out = pd.DataFrame(index=data.index)
        high = data["high"].astype(float)
        low = data["low"].astype(float)
        opening = data["open"].astype(float)
        close = data["close"].astype(float)
        valid = (
            high.gt(0.0) & low.gt(0.0) & opening.gt(0.0)
            & close.gt(0.0) & high.ge(low)
            & high.ge(opening) & high.ge(close)
            & low.le(opening) & low.le(close)
        )
        out["bar_valid"] = valid
        out["log_range"] = np.log(high.where(valid) / low.where(valid))
        out["price_range"] = (high - low).where(valid)
        out["signed_body"] = (close - opening).where(valid)
        out["fast_range"] = out["log_range"].rolling(
            params.fast_window, min_periods=params.fast_window
        ).mean()
        out["slow_range"] = out["log_range"].rolling(
            params.slow_window, min_periods=params.slow_window
        ).mean()
        out["range_ratio"] = out["fast_range"] / out["slow_range"].where(
            out["slow_range"].gt(0.0)
        )
        out["body_sum"] = out["signed_body"].rolling(
            params.pressure_window, min_periods=params.pressure_window
        ).sum()
        out["range_sum"] = out["price_range"].rolling(
            params.pressure_window, min_periods=params.pressure_window
        ).sum()
        out["body_efficiency"] = out["body_sum"] / out["range_sum"].where(
            out["range_sum"].gt(0.0)
        )
        out["squared_log_range"] = out["log_range"].pow(2)
        out["range_variance"] = out["squared_log_range"].rolling(
            params.fast_window, min_periods=params.fast_window
        ).mean() / (4.0 * np.log(2.0))
        out["annualized_volatility"] = np.sqrt(out["range_variance"] * 252.0)
        return out.replace([np.inf, -np.inf], np.nan)

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: RangeAccumulationParams,
    ) -> SignalFrame:
        # Both primitives must agree; losing either is the sole exit rule.
        entry_condition = (
            indicators["bar_valid"].fillna(False).astype(bool)
            & indicators["range_ratio"].lt(params.contraction_threshold)
            & indicators["body_efficiency"].gt(params.pressure_threshold)
        ).fillna(False)
        df = pd.DataFrame(index=data.index)
        df["signal"] = entry_condition.astype(int)
        volatility = indicators["annualized_volatility"].where(
            indicators["annualized_volatility"].gt(0.0)
        )
        exposure = (params.target_volatility / volatility).replace(
            [np.inf, -np.inf], np.nan
        ).clip(lower=params.min_size, upper=params.max_size).fillna(params.min_size)
        df["size"] = exposure.shift(1).fillna(params.min_size).astype(float)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")
