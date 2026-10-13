from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class FractionalImpulseParams:
    fractional_order: float = 0.65
    entry_strength: float = 0.8


class GeneratedStrategy(BaseStrategy[FractionalImpulseParams]):
    strategy_id = "gen_local-w2_1790989112"

    @classmethod
    def params_type(cls):
        return FractionalImpulseParams

    def warmup_bars(self, params: FractionalImpulseParams) -> int:
        return 15

    def indicators(self, data: pd.DataFrame, params: FractionalImpulseParams) -> pd.DataFrame:
        if not 0.0 < params.fractional_order < 1.0:
            raise ValueError("fractional_order must lie strictly between zero and one")
        if not np.isfinite(params.entry_strength) or params.entry_strength <= 0.0:
            raise ValueError("entry_strength must be finite and positive")

        close = data["close"].astype(float).replace([np.inf, -np.inf], np.nan)
        weights = [1.0]
        for k in range(1, 6):
            weights.append(-weights[-1] * (params.fractional_order - k + 1) / k)

        anchor = close.shift(6)
        fractional_close = pd.Series(0.0, index=data.index)
        for k, weight in enumerate(weights):
            fractional_close = fractional_close + weight * (close.shift(k) - anchor)

        impulse = fractional_close.rolling(3, min_periods=3).mean()
        magnitude = fractional_close.abs().rolling(3, min_periods=3).mean()
        coherence = impulse.abs().div(magnitude.replace(0.0, np.nan))
        change = close.diff()
        noise = change.abs().rolling(12, min_periods=12).mean()
        strength = impulse.div(noise.replace(0.0, np.nan)) * coherence

        return pd.DataFrame({
            "anchor": anchor,
            "fractional_close": fractional_close,
            "impulse": impulse,
            "magnitude": magnitude,
            "coherence": coherence,
            "change": change,
            "noise": noise,
            "strength": strength,
        }, index=data.index).replace([np.inf, -np.inf], np.nan)

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: FractionalImpulseParams,
    ) -> SignalFrame:
        strength = indicators["strength"].fillna(0.0)
        direction = np.where(
            strength >= params.entry_strength,
            1,
            np.where(strength <= -params.entry_strength, -1, 0),
        )
        signals = np.zeros(len(data), dtype=int)
        position = 0
        for i, candidate in enumerate(direction):
            if position == 0:
                position = int(candidate)
            elif candidate == -position:
                position = 0
            signals[i] = position

        df = pd.DataFrame({"signal": signals, "size": 1.0}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")
