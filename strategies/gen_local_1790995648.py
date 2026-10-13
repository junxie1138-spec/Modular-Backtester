from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class CompressionParams:
    range_window: int = 3
    gap_window: int = 10
    percentile_window: int = 126
    compression_percentile: float = 0.20
    holding_bars: int = 2


class GeneratedStrategy(BaseStrategy[CompressionParams]):
    strategy_id = "gen_local_1790995648"

    @classmethod
    def params_type(cls) -> type[CompressionParams]:
        return CompressionParams

    def warmup_bars(self, params: CompressionParams) -> int:
        return int(max(params.range_window, params.gap_window) + params.percentile_window + 1)

    def indicators(self, data: pd.DataFrame, params: CompressionParams) -> pd.DataFrame:
        previous_close = data["close"].shift(1).where(lambda s: s > 0)
        overnight_return = (data["open"] / previous_close - 1.0).replace([np.inf, -np.inf], np.nan)
        daytime_range = ((data["high"] - data["low"]) / previous_close).replace([np.inf, -np.inf], np.nan)
        daytime_range = daytime_range.where(daytime_range >= 0)
        range_rms = np.sqrt(daytime_range.pow(2).rolling(params.range_window, min_periods=params.range_window).mean())
        gap_std = overnight_return.rolling(params.gap_window, min_periods=params.gap_window).std(ddof=0)
        gap_mean = overnight_return.rolling(params.gap_window, min_periods=params.gap_window).mean()
        imbalance = (range_rms / gap_std.where(gap_std > 0)).replace([np.inf, -np.inf], np.nan)
        range_threshold = range_rms.shift(1).rolling(params.percentile_window, min_periods=params.percentile_window).quantile(params.compression_percentile)
        imbalance_threshold = imbalance.shift(1).rolling(params.percentile_window, min_periods=params.percentile_window).quantile(params.compression_percentile)
        return pd.DataFrame({
            "overnight_return": overnight_return,
            "daytime_range": daytime_range,
            "range_rms": range_rms,
            "gap_std": gap_std,
            "gap_mean": gap_mean,
            "imbalance": imbalance,
            "range_threshold": range_threshold,
            "imbalance_threshold": imbalance_threshold,
        }, index=data.index)

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: CompressionParams) -> SignalFrame:
        ready = indicators.notna().all(axis=1)
        entries = (
            ready
            & (indicators["range_rms"] < indicators["range_threshold"])
            & (indicators["imbalance"] < indicators["imbalance_threshold"])
            & (indicators["gap_mean"] > 0)
        ).fillna(False).to_numpy(dtype=bool)
        signals = np.zeros(len(data), dtype=np.int64)
        entry_bar = -1
        for bar in range(len(data)):
            if entry_bar >= 0:
                if bar - entry_bar >= params.holding_bars:
                    entry_bar = -1
                else:
                    signals[bar] = 1
            elif entries[bar]:
                entry_bar = bar
                signals[bar] = 1
        df = pd.DataFrame({"signal": signals, "size": 1.0}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")
