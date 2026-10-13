from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class VisibilityBridgeParams:
    topology_window: int = 18
    breakout_window: int = 9
    bridge_threshold: float = 0.16
    holding_bars: int = 4


class GeneratedStrategy(BaseStrategy[VisibilityBridgeParams]):
    strategy_id = "gen_local_1790937002"

    @classmethod
    def params_type(cls) -> type[VisibilityBridgeParams]:
        return VisibilityBridgeParams

    @staticmethod
    def warmup_bars(params: VisibilityBridgeParams) -> int:
        return int(max(params.topology_window, params.breakout_window + 1))

    @staticmethod
    def indicators(data: pd.DataFrame, params: VisibilityBridgeParams) -> pd.DataFrame:
        if params.topology_window < 8 or params.breakout_window < 2:
            raise ValueError("Topology window must be >= 8 and breakout window >= 2")
        if not np.isfinite(params.bridge_threshold) or not 0 < params.bridge_threshold <= 1:
            raise ValueError("Bridge threshold must be in (0, 1]")
        if not 3 <= params.holding_bars <= 5:
            raise ValueError("Holding bars must be between 3 and 5")

        out = pd.DataFrame(index=data.index)
        price = np.log(data["close"].where(data["close"] > 0))
        upper = pd.Series(0.0, index=data.index)
        lower = pd.Series(0.0, index=data.index)
        possible_weight = 0.0

        # Measure long chords crossing the boundary before the latest three bars.
        # A chord is visible above prices if its slope exceeds all nearer slopes.
        for recent_offset in range(3):
            endpoint = price.shift(recent_offset)
            highest_slope = pd.Series(-np.inf, index=data.index)
            lowest_slope = pd.Series(np.inf, index=data.index)
            for span in range(1, params.topology_window - recent_offset):
                slope = (price.shift(recent_offset + span) - endpoint) / span
                if span >= 5:
                    upper += slope.gt(highest_slope).astype(float) * span
                    lower += slope.lt(lowest_slope).astype(float) * span
                    possible_weight += span
                highest_slope = pd.Series(
                    np.maximum(highest_slope.to_numpy(), slope.to_numpy()),
                    index=data.index,
                )
                lowest_slope = pd.Series(
                    np.minimum(lowest_slope.to_numpy(), slope.to_numpy()),
                    index=data.index,
                )

        ready = price.rolling(
            params.topology_window, min_periods=params.topology_window
        ).count().eq(params.topology_window)
        out["upper_bridge_reach"] = (upper / possible_weight).where(ready)
        out["lower_bridge_reach"] = (lower / possible_weight).where(ready)
        out["bridge_asymmetry"] = out["upper_bridge_reach"] - out["lower_bridge_reach"]
        out["prior_channel_high"] = data["high"].shift(1).rolling(
            params.breakout_window, min_periods=params.breakout_window
        ).max()
        out["prior_channel_low"] = data["low"].shift(1).rolling(
            params.breakout_window, min_periods=params.breakout_window
        ).min()
        return out

    @staticmethod
    def generate_signals(
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: VisibilityBridgeParams,
    ) -> SignalFrame:
        long_candidate = (
            data["close"].gt(indicators["prior_channel_high"])
            & indicators["bridge_asymmetry"].ge(params.bridge_threshold)
        ).fillna(False)
        short_candidate = (
            data["close"].lt(indicators["prior_channel_low"])
            & indicators["bridge_asymmetry"].le(-params.bridge_threshold)
        ).fillna(False)
        confirmed_long = long_candidate & long_candidate.shift(1, fill_value=False)
        confirmed_short = short_candidate & short_candidate.shift(1, fill_value=False)
        entries = np.select(
            [confirmed_long.to_numpy(), confirmed_short.to_numpy()],
            [1, -1],
            default=0,
        ).astype(int)

        signals = np.zeros(len(data), dtype=int)
        position = 0
        exit_bar = -1
        for bar, entry in enumerate(entries):
            if position != 0:
                if bar == exit_bar:
                    position = 0
                    continue
            elif entry != 0:
                position = int(entry)
                exit_bar = bar + params.holding_bars
            signals[bar] = position

        df = pd.DataFrame(index=data.index)
        df["signal"] = signals
        df["size"] = 0.95
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")
