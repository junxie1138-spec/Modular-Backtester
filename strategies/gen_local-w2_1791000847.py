from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class GapEnvelopeParams:
    consensus_window: int = 18
    background_window: int = 63
    minimum_gap_breadth: float = 0.15
    maximum_envelope_ratio: float = 0.60


class GeneratedStrategy(BaseStrategy[GapEnvelopeParams]):
    strategy_id = "gen_local-w2_1791000847"

    @classmethod
    def params_type(cls) -> type[GapEnvelopeParams]:
        return GapEnvelopeParams

    @staticmethod
    def warmup_bars(params: GapEnvelopeParams) -> int:
        return int(max(params.background_window, params.consensus_window + 1))

    @staticmethod
    def indicators(data: pd.DataFrame, params: GapEnvelopeParams) -> pd.DataFrame:
        result = pd.DataFrame(index=data.index)
        previous_close = data["close"].shift(1).where(lambda values: values > 0.0)
        result["gap"] = (data["open"] / previous_close - 1.0).replace([np.inf, -np.inf], np.nan)
        result["gap_direction"] = np.sign(result["gap"])
        result["gap_breadth"] = result["gap_direction"].rolling(
            params.consensus_window, min_periods=params.consensus_window
        ).mean()
        result["monthly_high"] = data["high"].rolling(
            params.consensus_window, min_periods=params.consensus_window
        ).max()
        result["monthly_low"] = data["low"].rolling(
            params.consensus_window, min_periods=params.consensus_window
        ).min()
        result["background_high"] = data["high"].rolling(
            params.background_window, min_periods=params.background_window
        ).max()
        result["background_low"] = data["low"].rolling(
            params.background_window, min_periods=params.background_window
        ).min()
        result["monthly_span"] = result["monthly_high"] - result["monthly_low"]
        result["background_span"] = result["background_high"] - result["background_low"]
        result["envelope_ratio"] = (
            result["monthly_span"] / result["background_span"].where(result["background_span"] > 0.0)
        ).replace([np.inf, -np.inf], np.nan)
        return result

    @staticmethod
    def generate_signals(
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: GapEnvelopeParams,
    ) -> SignalFrame:
        valid = indicators[["gap_breadth", "envelope_ratio"]].notna().all(axis=1)
        gap_agrees = indicators["gap_breadth"] >= params.minimum_gap_breadth
        compression_agrees = indicators["envelope_ratio"] <= params.maximum_envelope_ratio
        entry_condition = valid & gap_agrees & compression_agrees
        df = pd.DataFrame(index=data.index)
        # Maintain exposure only while both entry predicates remain true.
        # A reversal of either predicate exits; there is no timed exit.
        df["signal"] = entry_condition.astype(int)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = 0.95
        return SignalFrame(data=df, signal_column="signal", size_column="size")
