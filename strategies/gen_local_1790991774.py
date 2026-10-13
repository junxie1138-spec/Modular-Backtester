from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class AlternatingRangeParams:
    anticorrelation_threshold: float = 0.30
    quiet_ratio: float = 0.85
    hold_bars: int = 18


class GeneratedStrategy(BaseStrategy[AlternatingRangeParams]):
    strategy_id = "gen_local_1790991774"

    @classmethod
    def params_type(cls) -> type[AlternatingRangeParams]:
        return AlternatingRangeParams

    def warmup_bars(self, params: AlternatingRangeParams) -> int:
        return 8

    def indicators(self, data: pd.DataFrame, params: AlternatingRangeParams) -> pd.DataFrame:
        out = pd.DataFrame(index=data.index)
        previous_close = data["close"].shift(1).replace(0.0, np.nan)
        true_range = pd.concat(
            [
                data["high"] - data["low"],
                (data["high"] - previous_close).abs(),
                (data["low"] - previous_close).abs(),
            ],
            axis=1,
        ).max(axis=1)
        out["normalized_range"] = (true_range / previous_close).replace([np.inf, -np.inf], np.nan)
        out["prior_range_mean"] = out["normalized_range"].rolling(6, min_periods=6).mean().shift(1)
        out["range_autocorrelation"] = out["normalized_range"].rolling(6, min_periods=6).corr(out["normalized_range"].shift(1)).replace([np.inf, -np.inf], np.nan)
        out["quiet_ratio"] = out["normalized_range"] / out["prior_range_mean"].replace(0.0, np.nan)
        out["drift"] = (data["close"] / data["close"].shift(6).replace(0.0, np.nan) - 1.0).replace([np.inf, -np.inf], np.nan)
        out["body_return"] = (data["close"] / data["open"].replace(0.0, np.nan) - 1.0).replace([np.inf, -np.inf], np.nan)
        return out.replace([np.inf, -np.inf], np.nan)

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: AlternatingRangeParams) -> SignalFrame:
        ready = indicators.notna().all(axis=1)
        entry = (
            ready
            & (indicators["range_autocorrelation"] < -params.anticorrelation_threshold)
            & (indicators["quiet_ratio"] < params.quiet_ratio)
            & (indicators["drift"] > 0.0)
            & (indicators["body_return"] > 0.0)
        ).fillna(False).to_numpy(dtype=bool)
        signal = np.zeros(len(data), dtype=np.int64)
        hold_bars = max(1, int(params.hold_bars))
        exit_bar = -1
        for i in range(len(data)):
            if exit_bar >= 0:
                if i < exit_bar:
                    signal[i] = 1
                else:
                    # A mandatory flat bar prevents immediate re-entry at expiry.
                    exit_bar = -1
            elif i >= self.warmup_bars(params) - 1 and entry[i]:
                signal[i] = 1
                exit_bar = i + hold_bars
        df = pd.DataFrame(index=data.index)
        df["signal"] = signal
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = 0.95
        return SignalFrame(data=df, signal_column="signal", size_column="size")
