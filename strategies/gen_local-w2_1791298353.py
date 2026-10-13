from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class AreaRecoveryParams:
    peak_window: int = 40
    volume_window: int = 20
    area_window: int = 5
    depth_threshold: float = 0.04
    area_threshold: float = 0.002
    hold_bars: int = 7


class GeneratedStrategy(BaseStrategy[AreaRecoveryParams]):
    strategy_id = "gen_local-w2_1791298353"

    @classmethod
    def params_type(cls) -> type[AreaRecoveryParams]:
        return AreaRecoveryParams

    @classmethod
    def warmup_bars(cls, params: AreaRecoveryParams) -> int:
        return max(params.peak_window, params.volume_window) + params.area_window + 2

    def indicators(self, data: pd.DataFrame, params: AreaRecoveryParams) -> pd.DataFrame:
        peak = data["close"].rolling(params.peak_window).max().replace(0, np.nan)
        depth = (1 - data["close"] / peak).clip(lower=0).fillna(0.0)
        mean_volume = data["volume"].rolling(params.volume_window).mean().replace(0, np.nan)
        relative_volume = (data["volume"] / mean_volume).fillna(0.0)
        depth_change = depth.diff().fillna(0.0)
        volume_change = relative_volume.diff().fillna(0.0)
        area_step = 0.5 * (
            depth_change.shift(1) * volume_change
            - volume_change.shift(1) * depth_change
        )
        area = area_step.rolling(params.area_window).sum().fillna(0.0)
        return pd.DataFrame(
            {"depth": depth, "depth_change": depth_change, "area": area},
            index=data.index,
        )

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: AreaRecoveryParams,
    ) -> SignalFrame:
        depth = indicators["depth"]
        change = indicators["depth_change"]
        area = indicators["area"]
        ready = pd.Series(np.arange(len(data)) >= self.warmup_bars(params), index=data.index)
        long_entry = (
            ready
            & (depth >= params.depth_threshold)
            & (area > params.area_threshold)
            & (area.shift(1) > params.area_threshold)
            & (change < 0)
            & (change.shift(1) < 0)
        ).fillna(False).to_numpy()
        short_entry = (
            ready
            & (depth >= params.depth_threshold)
            & (area < -params.area_threshold)
            & (area.shift(1) < -params.area_threshold)
            & (change > 0)
            & (change.shift(1) > 0)
        ).fillna(False).to_numpy()

        signal = np.zeros(len(data), dtype=int)
        side = 0
        remaining = 0
        for i in range(len(data)):
            if remaining > 0:
                signal[i] = side
                remaining -= 1
            elif side != 0:
                side = 0
            elif long_entry[i] or short_entry[i]:
                side = 1 if long_entry[i] else -1
                signal[i] = side
                remaining = params.hold_bars - 1

        df = pd.DataFrame({"signal": signal, "size": 1.0}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")
