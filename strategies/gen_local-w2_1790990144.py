from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class GeneratedParams:
    queue_window: int = 8
    reference_window: int = 126
    overflow_quantile: float = 0.75
    compression_quantile: float = 0.35
    size: float = 0.95


class GeneratedStrategy(BaseStrategy[GeneratedParams]):
    strategy_id = "gen_local-w2_1790990144"

    @classmethod
    def params_type(cls):
        return GeneratedParams

    def warmup_bars(self, params: GeneratedParams) -> int:
        return int(params.queue_window + params.reference_window + 2)

    def indicators(self, data: pd.DataFrame, params: GeneratedParams) -> pd.DataFrame:
        out = pd.DataFrame(index=data.index)
        previous_close = data["close"].shift(1).where(lambda s: s > 0)
        out["previous_close"] = previous_close
        out["gap"] = (data["open"] - previous_close) / previous_close
        out["intraday_return"] = (data["close"] - data["open"]) / previous_close
        out["normalized_range"] = ((data["high"] - data["low"]) / previous_close).where(data["high"] >= data["low"])

        # Count only the gap portion subsequently reversed during the session.
        out["gap_magnitude"] = out["gap"].abs()
        out["opposing_intraday_move"] = (-np.sign(out["gap"]) * out["intraday_return"]).clip(lower=0.0)
        out["rejected_gap"] = np.sign(out["gap"]) * out[["gap_magnitude", "opposing_intraday_move"]].min(axis=1, skipna=False)
        out["rejected_queue"] = out["rejected_gap"].rolling(params.queue_window).sum()
        out["range_capacity"] = out["normalized_range"].rolling(params.queue_window).sum()
        out["queue_load"] = out["rejected_queue"] / out["range_capacity"].replace(0.0, np.nan)
        out["absolute_load"] = out["queue_load"].abs()
        out["mean_range"] = out["normalized_range"].rolling(params.queue_window).mean()

        # Thresholds exclude the current observation.
        out["overflow_threshold"] = out["absolute_load"].rolling(params.reference_window).quantile(params.overflow_quantile).shift(1)
        out["compression_threshold"] = out["mean_range"].rolling(params.reference_window).quantile(params.compression_quantile).shift(1)
        return out.replace([np.inf, -np.inf], np.nan)

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: GeneratedParams,
    ) -> SignalFrame:
        required = ["queue_load", "absolute_load", "mean_range", "overflow_threshold", "compression_threshold"]
        valid = indicators[required].notna().all(axis=1)
        compressed = indicators["mean_range"] <= indicators["compression_threshold"]
        overloaded = indicators["absolute_load"] > indicators["overflow_threshold"]
        eligible = valid & compressed & overloaded
        candidates = np.where(
            eligible.to_numpy(),
            -np.sign(indicators["queue_load"].fillna(0.0).to_numpy()),
            0,
        ).astype(int)

        raw_signal = np.zeros(len(data), dtype=int)
        state = 0
        for i, candidate in enumerate(candidates):
            if state == 0:
                state = int(candidate)
            elif candidate == -state:
                # Opposite qualifying entry condition exits to flat.
                # A subsequent qualifying bar can establish the new direction.
                state = 0
            raw_signal[i] = state

        df = pd.DataFrame(index=data.index)
        df["signal"] = raw_signal
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = float(params.size)
        return SignalFrame(data=df, signal_column="signal", size_column="size")
