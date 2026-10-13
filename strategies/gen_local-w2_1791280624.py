from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class RangeAlternationParams:
    autocorr_window: int = 30
    min_reversal: float = 0.25
    min_contraction: float = 0.12
    holding_bars: int = 8
    min_size: float = 0.25
    max_size: float = 0.95


class GeneratedStrategy(BaseStrategy[RangeAlternationParams]):
    strategy_id = "gen_local-w2_1791280624"

    @classmethod
    def params_type(cls) -> type[RangeAlternationParams]:
        return RangeAlternationParams

    def warmup_bars(self, params: RangeAlternationParams) -> int:
        return int(max(2, params.autocorr_window) + 2)

    def indicators(self, data: pd.DataFrame, params: RangeAlternationParams) -> pd.DataFrame:
        window = max(2, int(params.autocorr_window))
        close = data["close"].where(data["close"] > 0)
        span = (data["high"] - data["low"]).where(lambda x: x > 0)
        normalized_range = span / close
        log_range = np.log(normalized_range)
        innovation = log_range.diff()
        lagged_innovation = innovation.shift(1)
        autocorrelation = innovation.rolling(window, min_periods=window).corr(lagged_innovation)
        close_location = (close - data["low"]) / span
        momentum = close / close.shift(5) - 1.0
        out = pd.DataFrame({
            "normalized_range": normalized_range,
            "log_range": log_range,
            "innovation": innovation,
            "lagged_innovation": lagged_innovation,
            "autocorrelation": autocorrelation,
            "close_location": close_location,
            "momentum": momentum,
        }, index=data.index)
        return out.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: RangeAlternationParams) -> SignalFrame:
        if params.holding_bars < 1:
            raise ValueError("holding_bars must be positive")
        if not 0.0 < params.min_reversal < 1.0:
            raise ValueError("min_reversal must be between zero and one")
        if params.min_contraction <= 0.0:
            raise ValueError("min_contraction must be positive")
        if not 0.0 < params.min_size <= params.max_size <= 1.0:
            raise ValueError("sizes must satisfy 0 < min_size <= max_size <= 1")

        corr = indicators["autocorrelation"]
        innovation = indicators["innovation"]
        eligible = (
            (corr <= -params.min_reversal)
            & (innovation <= -params.min_contraction)
            & (indicators["close_location"] >= 0.70)
            & (indicators["momentum"] > 0.0)
        )
        eligible.iloc[:self.warmup_bars(params)] = False
        reversal_strength = (-corr).clip(0.0, 1.0)
        contraction_strength = (-innovation / (3.0 * params.min_contraction)).clip(0.0, 1.0)
        strength = np.sqrt(reversal_strength * contraction_strength)
        entry_sizes = params.min_size + (params.max_size - params.min_size) * strength

        count = len(data)
        signals = np.zeros(count, dtype=int)
        sizes = np.full(count, params.min_size, dtype=float)
        candidates = eligible.to_numpy(dtype=bool)
        candidate_sizes = entry_sizes.to_numpy(dtype=float)
        exit_bar = -1
        active_size = params.min_size
        holding = int(params.holding_bars)

        for i in range(count):
            if exit_bar >= 0:
                if i < exit_bar:
                    signals[i] = 1
                    sizes[i] = active_size
                    continue
                exit_bar = -1
                continue
            if candidates[i]:
                active_size = float(candidate_sizes[i])
                signals[i] = 1
                sizes[i] = active_size
                exit_bar = i + holding

        df = pd.DataFrame({"signal": signals, "size": sizes}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = df["size"].shift(1).fillna(params.min_size).astype(float)
        return SignalFrame(data=df, signal_column="signal", size_column="size")
