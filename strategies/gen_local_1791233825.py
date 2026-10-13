from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class OrdinalChangeParams:
    window: int = 4
    complexity_drop: float = 0.10
    bullish_fraction: float = 0.75


class GeneratedStrategy(BaseStrategy[OrdinalChangeParams]):
    strategy_id = "gen_local_1791233825"

    @classmethod
    def params_type(cls) -> type[OrdinalChangeParams]:
        return OrdinalChangeParams

    @staticmethod
    def warmup_bars(params: OrdinalChangeParams) -> int:
        return 2 * min(4, max(2, int(params.window))) + 2

    @staticmethod
    def indicators(data: pd.DataFrame, params: OrdinalChangeParams) -> pd.DataFrame:
        window = min(4, max(2, int(params.window)))
        out = pd.DataFrame(index=data.index)
        a = data["close"].shift(2)
        b = data["close"].shift(1)
        c = data["close"]
        valid = np.isfinite(a) & np.isfinite(b) & np.isfinite(c)
        # Strict comparisons break equal-price ranks by chronological order.
        out["ordinal_code"] = (
            4 * (a > b).astype(int)
            + 2 * (b > c).astype(int)
            + (a > c).astype(int)
        ).astype(float).where(valid)
        concentration = pd.Series(0.0, index=data.index)
        for code in (0, 2, 3, 4, 5, 7):
            probability = (out["ordinal_code"] == code).astype(float).where(valid)
            out[f"motif_{code}_frequency"] = probability.rolling(
                window, min_periods=window
            ).mean()
            concentration = concentration + out[f"motif_{code}_frequency"].pow(2)
        out["complexity"] = 1.0 - concentration
        out["previous_complexity"] = out["complexity"].shift(window)
        out["bullish_fraction"] = (c > a).astype(float).where(valid).rolling(
            window, min_periods=window
        ).mean()
        out["previous_bullish_fraction"] = out["bullish_fraction"].shift(window)
        return out

    @staticmethod
    def generate_signals(
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: OrdinalChangeParams,
    ) -> SignalFrame:
        required = [
            "complexity", "previous_complexity",
            "bullish_fraction", "previous_bullish_fraction",
        ]
        ready = indicators[required].notna().all(axis=1)
        condition = (
            ready
            & (indicators["previous_complexity"] - indicators["complexity"] >= params.complexity_drop)
            & (indicators["bullish_fraction"] >= params.bullish_fraction)
            & (indicators["previous_bullish_fraction"] <= 0.5)
        )
        df = pd.DataFrame(index=data.index)
        # The position exits only when this same entry condition flips false.
        df["signal"] = condition.fillna(False).astype(int)
        df["size"] = 1.0
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")
