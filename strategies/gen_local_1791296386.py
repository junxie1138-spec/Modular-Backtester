from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class CompressionDrawdownParams:
    range_bars: int = 5
    peak_bars: int = 60
    atr_bars: int = 20
    compression_atr: float = 2.5
    deep_atr: float = 2.0
    trail_atr: float = 1.5
    hold_bars: int = 4


class GeneratedStrategy(BaseStrategy[CompressionDrawdownParams]):
    strategy_id = "gen_local_1791296386"

    @classmethod
    def params_type(cls):
        return CompressionDrawdownParams

    def warmup_bars(self, params: CompressionDrawdownParams) -> int:
        return max(params.peak_bars, params.atr_bars + 1, params.range_bars + 1)

    def indicators(self, data: pd.DataFrame, params: CompressionDrawdownParams) -> pd.DataFrame:
        previous_close = data["close"].shift(1)
        true_range = pd.concat(
            [
                data["high"] - data["low"],
                (data["high"] - previous_close).abs(),
                (data["low"] - previous_close).abs(),
            ],
            axis=1,
        ).max(axis=1)
        atr = true_range.rolling(params.atr_bars, min_periods=params.atr_bars).mean().fillna(0.0)
        prior_high = data["high"].rolling(params.range_bars, min_periods=params.range_bars).max().shift(1)
        prior_low = data["low"].rolling(params.range_bars, min_periods=params.range_bars).min().shift(1)
        peak_close = data["close"].rolling(params.peak_bars, min_periods=params.peak_bars).max()
        depth = ((peak_close - data["close"]) / atr.replace(0.0, np.nan)).replace([np.inf, -np.inf], np.nan)
        return pd.DataFrame(
            {
                "atr": atr,
                "prior_high": prior_high.fillna(data["high"]),
                "prior_low": prior_low.fillna(data["low"]),
                "depth_atr": depth.fillna(0.0),
            },
            index=data.index,
        )

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: CompressionDrawdownParams,
    ) -> SignalFrame:
        signals = np.zeros(len(data), dtype=int)
        sizes = np.ones(len(data), dtype=float)
        close = data["close"].to_numpy(dtype=float)
        high = data["high"].to_numpy(dtype=float)
        low = data["low"].to_numpy(dtype=float)
        atr = indicators["atr"].to_numpy(dtype=float)
        prior_high = indicators["prior_high"].to_numpy(dtype=float)
        prior_low = indicators["prior_low"].to_numpy(dtype=float)
        depth = indicators["depth_atr"].to_numpy(dtype=float)
        side = 0
        age = 0
        entry_size = 1.0
        high_water = low_water = stop = 0.0

        for i in range(len(data)):
            if side:
                age += 1
                if side > 0:
                    high_water = max(high_water, close[i])
                    stop = max(stop, high_water - params.trail_atr * atr[i])
                    exit_now = close[i] <= stop
                else:
                    low_water = min(low_water, close[i])
                    stop = min(stop, low_water + params.trail_atr * atr[i])
                    exit_now = close[i] >= stop
                if exit_now or age >= params.hold_bars:
                    side = 0
                    continue
                signals[i] = side
                sizes[i] = entry_size
                continue

            if i < self.warmup_bars(params) or atr[i] <= 0:
                continue
            if prior_high[i] - prior_low[i] > params.compression_atr * atr[i]:
                continue
            if low[i] < prior_low[i] < close[i] and depth[i] >= params.deep_atr:
                side = 1
                entry_size = float(np.clip(depth[i] / (2.0 * params.deep_atr), 0.25, 1.0))
                high_water = close[i]
                stop = high_water - params.trail_atr * atr[i]
            elif high[i] > prior_high[i] > close[i] and depth[i] <= params.deep_atr / 2.0:
                side = -1
                entry_size = float(np.clip(1.0 - depth[i] / params.deep_atr, 0.25, 1.0))
                low_water = close[i]
                stop = low_water + params.trail_atr * atr[i]
            if side:
                age = 0
                signals[i] = side
                sizes[i] = entry_size

        df = pd.DataFrame({"signal": signals, "size": sizes}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = df["size"].shift(1).fillna(1.0).astype(float)
        return SignalFrame(data=df, signal_column="signal", size_column="size")
