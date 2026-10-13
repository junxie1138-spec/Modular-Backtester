from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class OrdinalMemoryParams:
    memory_window: int = 63
    baseline_window: int = 63
    impulse_window: int = 8
    pullback_window: int = 10
    volatility_window: int = 20
    complexity_split: float = 0.68
    minimum_dip: float = 0.35


class GeneratedStrategy(BaseStrategy[OrdinalMemoryParams]):
    strategy_id = "gen_local-w2_1790997222"

    @classmethod
    def params_type(cls):
        return OrdinalMemoryParams

    def warmup_bars(self, params: OrdinalMemoryParams) -> int:
        return int(max(
            max(6, params.memory_window) + 3,
            max(2, params.baseline_window),
            max(2, params.impulse_window) + 1,
            max(2, params.pullback_window),
            max(2, params.volatility_window) + 1,
        ))

    def indicators(self, data: pd.DataFrame, params: OrdinalMemoryParams) -> pd.DataFrame:
        out = pd.DataFrame(index=data.index)
        close = data["close"].astype(float).where(data["close"] > 0)
        returns = close.pct_change(fill_method=None).replace([np.inf, -np.inf], np.nan)
        a, b, c = returns.shift(2), returns.shift(1), returns
        valid = a.notna() & b.notna() & c.notna()
        pattern = ((a > b).astype(int) + 2 * (a > c).astype(int)
                   + 4 * (b > c).astype(int)).astype(float).where(valid)
        previous = pattern.shift(1)
        pair_valid = pattern.notna() & previous.notna()
        window = max(6, params.memory_window)
        joint_entropy = pd.Series(0.0, index=data.index)
        previous_entropy = pd.Series(0.0, index=data.index)
        codes = (0, 1, 3, 4, 6, 7)
        for left in codes:
            probability = previous.eq(left).astype(float).where(pair_valid).rolling(
                window, min_periods=window
            ).mean()
            previous_entropy = previous_entropy - probability * np.log(
                probability.where(probability > 0, 1.0)
            )
            for right in codes:
                probability = (previous.eq(left) & pattern.eq(right)).astype(float).where(
                    pair_valid
                ).rolling(window, min_periods=window).mean()
                joint_entropy = joint_entropy - probability * np.log(
                    probability.where(probability > 0, 1.0)
                )
        out["returns"] = returns
        out["pattern"] = pattern
        out["previous_pattern"] = previous
        out["joint_entropy"] = joint_entropy
        out["previous_entropy"] = previous_entropy
        out["complexity"] = ((joint_entropy - previous_entropy) / np.log(6.0)).clip(0, 1)
        out["baseline"] = close.rolling(max(2, params.baseline_window)).mean()
        out["volatility"] = returns.rolling(max(2, params.volatility_window)).std(ddof=0)
        denominator = out["volatility"].where(out["volatility"] > 0)
        impulse_window = max(2, params.impulse_window)
        out["impulse"] = returns.rolling(impulse_window).sum() / (
            denominator * np.sqrt(float(impulse_window))
        )
        out["pullback_mean"] = close.rolling(max(2, params.pullback_window)).mean()
        out["dip"] = (out["pullback_mean"] - close) / (close * denominator)
        return out.replace([np.inf, -np.inf], np.nan)

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: OrdinalMemoryParams,
    ) -> SignalFrame:
        ready = indicators[["complexity", "baseline", "impulse", "dip"]].notna().all(axis=1)
        ordered = indicators["complexity"] < params.complexity_split
        trend_entry = ordered & indicators["impulse"].gt(0)
        pullback_entry = (~ordered) & indicators["dip"].gt(params.minimum_dip)
        entry = ready & data["close"].gt(indicators["baseline"]) & (trend_entry | pullback_entry)
        # The same regime-dependent entry predicate controls reversal exits.
        strength = indicators["impulse"].div(2.0).where(
            ordered, indicators["dip"].div(2.0)
        ).clip(0.0, 1.0).fillna(0.0)
        df = pd.DataFrame(index=data.index)
        df["signal"] = entry.astype(int)
        df["size"] = (0.20 + 0.75 * strength).astype(float)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = df["size"].shift(1).fillna(0.20).astype(float)
        return SignalFrame(data=df, signal_column="signal", size_column="size")
