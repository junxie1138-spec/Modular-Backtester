from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class TidalRankParams:
    tide_window: int = 126
    wave_window: int = 40
    mom_bars: int = 10
    regime_thresh: float = 0.50
    entry_pct: float = 0.30
    pct_window: int = 252


class GeneratedStrategy(BaseStrategy[TidalRankParams]):
    strategy_id = "gen_jay_1778884496"

    @classmethod
    def params_type(cls):
        return TidalRankParams

    @staticmethod
    def warmup_bars(params: TidalRankParams) -> int:
        return max(params.pct_window + params.wave_window + params.mom_bars, params.tide_window) + 1

    @staticmethod
    def indicators(data: pd.DataFrame, params: TidalRankParams) -> pd.DataFrame:
        close = data["close"]

        # Long-term tide level: where is price within its recent range?
        close_rank = close.rolling(params.tide_window).rank(pct=True)

        # Short-term standing wave: momentum rank within its recent distribution
        mom = close.pct_change(params.mom_bars)
        wave_rank = mom.rolling(params.wave_window).rank(pct=True)

        # Dynamic entry threshold: rolling quantile of the wave rank itself.
        # This is the percentile-threshold twist — the gate adapts to the
        # historical distribution of wave_rank rather than using a fixed level.
        dynamic_thresh = wave_rank.rolling(params.pct_window).quantile(params.entry_pct)

        ind = pd.DataFrame(index=data.index)
        ind["close_rank"] = close_rank
        ind["wave_rank"] = wave_rank
        ind["dynamic_thresh"] = dynamic_thresh
        return ind

    @staticmethod
    def generate_signals(
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: TidalRankParams,
    ) -> SignalFrame:
        df = pd.DataFrame(index=data.index)

        # High-tide regime: price is in the upper half of its rolling window
        high_tide = indicators["close_rank"] > params.regime_thresh

        # Wave trough: momentum rank has fallen below its own dynamic percentile
        at_trough = indicators["wave_rank"] < indicators["dynamic_thresh"]

        # Entry: bullish regime AND at a standing-wave trough.
        # Signal-reversal exit is implicit: position is held exactly while
        # both conditions are true and closes as soon as either flips.
        raw_signal = (high_tide & at_trough).astype(int)

        df["signal"] = raw_signal.shift(1).fillna(0).astype(int)
        df["size"] = 1.0

        return SignalFrame(data=df, signal_column="signal", size_column="size")
