from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class CompressionParams:
    z_threshold: float = 1.15
    compression_threshold: float = 0.80
    profit_target: float = 0.012
    max_hold_bars: int = 8


class GeneratedStrategy(BaseStrategy[CompressionParams]):
    strategy_id = "gen_local_1791000479"

    @classmethod
    def params_type(cls) -> type[CompressionParams]:
        return CompressionParams

    def warmup_bars(self, params: CompressionParams) -> int:
        return 10

    def indicators(self, data: pd.DataFrame, params: CompressionParams) -> pd.DataFrame:
        close = data["close"].astype(float)
        mean = close.rolling(8, min_periods=8).mean()
        deviation = close.rolling(8, min_periods=8).std(ddof=0)
        z = ((close - mean) / deviation.where(deviation > 0.0)).replace([np.inf, -np.inf], np.nan)
        relative_range = (data["high"] - data["low"]) / close.where(close > 0.0)
        recent_range = relative_range.rolling(3, min_periods=3).mean()
        baseline_range = relative_range.rolling(8, min_periods=8).mean()
        compression = (recent_range / baseline_range.where(baseline_range > 0.0)).replace([np.inf, -np.inf], np.nan)
        return pd.DataFrame({
            "mean": mean,
            "deviation": deviation,
            "z": z,
            "z_previous": z.shift(1),
            "z_previous_two": z.shift(2),
            "relative_range": relative_range,
            "recent_range": recent_range,
            "baseline_range": baseline_range,
            "compression": compression,
        }, index=data.index)

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: CompressionParams) -> SignalFrame:
        z = indicators["z"]
        previous = indicators["z_previous"]
        previous_two = indicators["z_previous_two"]
        valid = indicators[["z", "z_previous", "z_previous_two", "compression"]].notna().all(axis=1)
        compressed = valid & (indicators["compression"] <= params.compression_threshold)
        long_entry = compressed & (previous <= -params.z_threshold) & (previous_two <= -params.z_threshold) & (z > previous) & (z < -0.5)
        short_entry = compressed & (previous >= params.z_threshold) & (previous_two >= params.z_threshold) & (z < previous) & (z > 0.5)
        candidates = np.select([long_entry, short_entry], [1, -1], default=0).astype(int)
        opens = data["open"].to_numpy(dtype=float)
        closes = data["close"].to_numpy(dtype=float)
        decisions = np.zeros(len(data), dtype=int)
        position = 0
        entry_price = np.nan
        entry_bar = -1
        hold_limit = max(1, int(params.max_hold_bars))
        target = max(0.0, float(params.profit_target))

        for i in range(len(data)):
            if position != 0:
                if i == entry_bar:
                    entry_price = opens[i]
                    if not np.isfinite(entry_price) or entry_price <= 0.0:
                        position = 0
                        entry_price = np.nan
                        decisions[i] = 0
                        continue
                elapsed = i - entry_bar + 1
                gain = position * (closes[i] / entry_price - 1.0) if np.isfinite(closes[i]) else np.nan
                if (np.isfinite(gain) and gain >= target) or elapsed >= hold_limit:
                    position = 0
                    entry_price = np.nan
            elif candidates[i] != 0:
                position = int(candidates[i])
                entry_bar = i + 1
                entry_price = np.nan
            decisions[i] = position

        df = pd.DataFrame({"signal": decisions, "size": 0.95}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")
