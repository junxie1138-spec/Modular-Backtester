from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class GeneratedParams:
    slope_span: int = 4
    baseline_span: int = 9
    weak_threshold: float = 0.15


class GeneratedStrategy(BaseStrategy[GeneratedParams]):
    strategy_id = "gen_local-w2_1791298087"

    @classmethod
    def params_type(cls):
        return GeneratedParams

    @classmethod
    def warmup_bars(cls, params: GeneratedParams) -> int:
        return max(205, params.slope_span + params.baseline_span)

    def indicators(self, data: pd.DataFrame, params: GeneratedParams) -> pd.DataFrame:
        out = pd.DataFrame(index=data.index)
        out["ma200"] = data["close"].rolling(200, min_periods=200).mean()
        out["ma_rising"] = out["ma200"].diff(5) > 0
        for price in ("close", "low"):
            log_return = np.log(data[price].clip(lower=1e-12)).diff()
            slope = log_return.ewm(span=params.slope_span, adjust=False, min_periods=params.slope_span).mean()
            baseline = slope.ewm(span=params.baseline_span, adjust=False, min_periods=params.baseline_span).mean()
            out[f"{price}_slope"] = slope
            out[f"{price}_innovation"] = slope - baseline
        out["threshold"] = np.log(data["close"].clip(lower=1e-12)).diff().abs().ewm(
            span=params.baseline_span, adjust=False, min_periods=params.baseline_span
        ).mean() * params.weak_threshold
        return out

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: GeneratedParams,
    ) -> SignalFrame:
        above = data["close"] > indicators["ma200"]
        rising = indicators["ma_rising"]
        defended_dip = rising & (indicators["low_innovation"] > 0) & (indicators["close_slope"] > 0)
        recovery = (~rising) & (indicators["close_innovation"] > indicators["threshold"]) & (indicators["low_slope"] > 0)
        entry_condition = (above & (defended_dip | recovery)).fillna(False)
        df = pd.DataFrame(index=data.index)
        df["signal"] = entry_condition.astype(int)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = 1.0
        return SignalFrame(data=df, signal_column="signal", size_column="size")
