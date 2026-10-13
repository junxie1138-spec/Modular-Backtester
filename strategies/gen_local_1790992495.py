from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class DrawdownPropagationParams:
    peak_window: int = 20
    volatility_window: int = 10
    regime_window: int = 8
    stress_fraction: float = 0.625
    minimum_depth: float = 0.9
    calm_depth_ceiling: float = 3.0
    minimum_size: float = 0.2
    maximum_size: float = 0.95


class GeneratedStrategy(BaseStrategy[DrawdownPropagationParams]):
    strategy_id = "gen_local_1790992495"

    @classmethod
    def params_type(cls) -> type[DrawdownPropagationParams]:
        return DrawdownPropagationParams

    @staticmethod
    def warmup_bars(params: DrawdownPropagationParams) -> int:
        return int(max(params.peak_window + params.regime_window + 2,
                       params.volatility_window + 1))

    @staticmethod
    def indicators(data: pd.DataFrame, params: DrawdownPropagationParams) -> pd.DataFrame:
        close = data["close"]
        previous_close = close.shift(1)
        true_range = pd.concat([
            data["high"] - data["low"],
            (data["high"] - previous_close).abs(),
            (data["low"] - previous_close).abs(),
        ], axis=1).max(axis=1).where(previous_close.notna())
        atr = true_range.rolling(params.volatility_window,
                                 min_periods=params.volatility_window).mean()
        peak = close.rolling(params.peak_window,
                             min_periods=params.peak_window).max()
        depth = (peak - close).clip(lower=0.0)
        depth_change = depth.diff()
        prior_change = depth_change.shift(1)
        expanding = (depth_change > 0.0).astype(float).where(depth_change.notna())
        prevalence = expanding.rolling(params.regime_window,
                                        min_periods=params.regime_window).mean()
        depth_units = depth / atr.where(atr > 0.0)
        out = pd.DataFrame({
            "peak": peak,
            "atr": atr,
            "depth": depth,
            "depth_change": depth_change,
            "prior_change": prior_change,
            "prevalence": prevalence,
            "depth_units": depth_units,
        }, index=data.index)
        finite = out.replace([np.inf, -np.inf], np.nan)
        finite["ready"] = finite.notna().all(axis=1).astype(float)
        return finite.fillna(0.0)

    @staticmethod
    def generate_signals(data: pd.DataFrame, indicators: pd.DataFrame,
                         ctx: StrategyContext,
                         params: DrawdownPropagationParams) -> SignalFrame:
        stress = indicators["prevalence"] >= params.stress_fraction
        fresh_expansion = ((indicators["depth_change"] > 0.0)
                           & (indicators["prior_change"] <= 0.0))
        first_contraction = ((indicators["depth_change"] < 0.0)
                             & (indicators["prior_change"] >= 0.0))
        calm_entry = (~stress & fresh_expansion
                      & (indicators["depth_units"] <= params.calm_depth_ceiling))
        stressed_entry = stress & first_contraction
        entry_condition = ((indicators["ready"] > 0.0)
                           & (indicators["depth_units"] >= params.minimum_depth)
                           & (calm_entry | stressed_entry))
        # Retain exposure only while the entry predicate remains true;
        # its reversal is the sole exit mechanic.
        df = pd.DataFrame(index=data.index)
        df["signal"] = entry_condition.astype(int)
        agreement = indicators["prevalence"].where(
            stress, 1.0 - indicators["prevalence"]
        ).clip(0.0, 1.0)
        depth_strength = indicators["depth_units"].clip(lower=0.0)
        depth_strength = depth_strength / (1.0 + depth_strength)
        lower_size = float(np.clip(params.minimum_size, 0.001, 1.0))
        upper_size = float(np.clip(params.maximum_size, lower_size, 1.0))
        decision_size = lower_size + (upper_size - lower_size) * depth_strength * agreement
        df["size"] = decision_size.shift(1).fillna(lower_size).astype(float)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")
