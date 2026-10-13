from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class RangeEscapeParams:
    atr_window: int = 4
    breakout_atr: float = 0.15
    minimum_overlap: float = 0.55


class GeneratedStrategy(BaseStrategy[RangeEscapeParams]):
    strategy_id = "gen_local-w2_1790994514"

    @classmethod
    def params_type(cls) -> type[RangeEscapeParams]:
        return RangeEscapeParams

    def warmup_bars(self, params: RangeEscapeParams) -> int:
        return int(np.clip(params.atr_window, 2, 8)) + 2

    def indicators(self, data: pd.DataFrame, params: RangeEscapeParams) -> pd.DataFrame:
        window = int(np.clip(params.atr_window, 2, 8))
        previous_close = data["close"].shift(1)
        true_range = pd.concat([
            data["high"] - data["low"],
            (data["high"] - previous_close).abs(),
            (data["low"] - previous_close).abs(),
        ], axis=1).max(axis=1)
        prior_atr = true_range.rolling(window, min_periods=window).mean().shift(1)
        high_one = data["high"].shift(1)
        high_two = data["high"].shift(2)
        low_one = data["low"].shift(1)
        low_two = data["low"].shift(2)
        overlap_width = (
            pd.concat([high_one, high_two], axis=1).min(axis=1, skipna=False)
            - pd.concat([low_one, low_two], axis=1).max(axis=1, skipna=False)
        ).clip(lower=0.0)
        smaller_range = pd.concat([
            high_one - low_one, high_two - low_two
        ], axis=1).min(axis=1, skipna=False).replace(0.0, np.nan)
        overlap = (overlap_width / smaller_range).clip(0.0, 1.0)
        ceiling = pd.concat([high_one, high_two], axis=1).max(axis=1, skipna=False)
        escape = (data["close"] - ceiling) / prior_atr.replace(0.0, np.nan)
        expansion = true_range / prior_atr.replace(0.0, np.nan)
        result = pd.DataFrame({
            "true_range": true_range,
            "prior_atr": prior_atr,
            "overlap_width": overlap_width,
            "smaller_range": smaller_range,
            "overlap": overlap,
            "ceiling": ceiling,
            "escape_atr": escape,
            "range_expansion": expansion,
        }, index=data.index).replace([np.inf, -np.inf], np.nan)
        result["ready"] = result.notna().all(axis=1) & result["prior_atr"].gt(0.0)
        return result.fillna(0.0)

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: RangeEscapeParams,
    ) -> SignalFrame:
        entry_condition = (
            indicators["ready"].astype(bool)
            & indicators["overlap"].ge(float(np.clip(params.minimum_overlap, 0.0, 1.0)))
            & indicators["escape_atr"].gt(max(0.0, float(params.breakout_atr)))
            & indicators["range_expansion"].gt(1.0)
        )
        # Remain long only while the full entry condition holds; its reversal exits.
        df = pd.DataFrame(index=data.index)
        df["signal"] = entry_condition.astype(int)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = 0.95
        assert df["signal"].isin([0, 1]).all() and df["size"].gt(0.0).all()
        return SignalFrame(data=df, signal_column="signal", size_column="size")
