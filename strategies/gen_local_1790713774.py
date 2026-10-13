from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class OrdinalComplexityParams:
    pattern_window: int = 20
    trend_window: int = 20
    complexity_threshold: float = 0.62
    hold_bars: int = 15
    min_size: float = 0.25
    max_size: float = 1.0


class GeneratedStrategy(BaseStrategy[OrdinalComplexityParams]):
    strategy_id = "gen_local_1790713774"

    @classmethod
    def params_type(cls):
        return OrdinalComplexityParams

    def warmup_bars(self, params: OrdinalComplexityParams) -> int:
        return max(params.pattern_window + 2, params.trend_window)

    def indicators(self, data: pd.DataFrame, params: OrdinalComplexityParams) -> pd.DataFrame:
        returns = data["close"].pct_change()
        # Rank each three-return sequence into one of its six ordinal patterns.
        a, b, c = returns, returns.shift(1), returns.shift(2)
        pattern = (
            (a > b).astype(int) * 4 + (a > c).astype(int) * 2 + (b > c).astype(int)
        )
        valid = returns.notna() & b.notna() & c.notna()
        counts = pd.concat(
            [(pattern.eq(k) & valid).astype(float).rolling(params.pattern_window).sum() for k in range(6)],
            axis=1,
        ).fillna(0.0)
        probs = counts.div(counts.sum(axis=1).replace(0.0, np.nan), axis=0)
        entropy = -(probs * np.log(probs.where(probs > 0.0))).sum(axis=1) / np.log(6.0)
        out = pd.DataFrame(index=data.index)
        out["complexity"] = entropy.fillna(0.0)
        out["trend"] = data["close"].rolling(params.trend_window).mean()
        return out

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: OrdinalComplexityParams,
    ) -> SignalFrame:
        raw = (
            (indicators["complexity"] >= params.complexity_threshold)
            & (data["close"] > indicators["trend"])
            & indicators["trend"].notna()
        ).fillna(False)
        signal = np.zeros(len(data), dtype=int)
        size = np.full(len(data), params.min_size, dtype=float)
        active_until = -1
        active_size = params.min_size
        for i, entry in enumerate(raw.to_numpy()):
            if i <= active_until:
                signal[i] = 1
                size[i] = active_size
            elif entry:
                signal[i] = 1
                active_until = i + params.hold_bars - 1
                strength = float(indicators["complexity"].iloc[i])
                scale = np.clip((strength - params.complexity_threshold) / max(1.0 - params.complexity_threshold, 1e-9), 0.0, 1.0)
                active_size = params.min_size + scale * (params.max_size - params.min_size)
                size[i] = active_size
        df = pd.DataFrame(index=data.index)
        df["signal"] = pd.Series(signal, index=data.index).shift(1).fillna(0).astype(int)
        df["size"] = pd.Series(size, index=data.index).shift(1).fillna(params.min_size).astype(float)
        return SignalFrame(data=df, signal_column="signal", size_column="size")
