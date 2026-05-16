from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class Params:
    lookback: int = 63
    smooth: int = 5


class GeneratedStrategy(BaseStrategy[Params]):
    strategy_id = "gen_jay_1778908115"

    @classmethod
    def params_type(cls) -> type[Params]:
        return Params

    @staticmethod
    def warmup_bars(params: Params) -> int:
        # lookback for HWM, +2 for two sequential diffs, +smooth for curvature smoothing
        return params.lookback + params.smooth + 3

    @staticmethod
    def indicators(data: pd.DataFrame, params: Params) -> pd.DataFrame:
        ind = pd.DataFrame(index=data.index)
        close = data["close"]

        # Rolling high-water mark and fractional drawdown (0 at peak, negative below)
        hwm = close.rolling(params.lookback, min_periods=params.lookback).max()
        drawdown = (close - hwm) / hwm

        # Stressed regime: current drawdown is worse than its expanding historical median
        # expanding(min_periods=1) skips NaN bars so no double-window blowup
        dd_hist_median = drawdown.expanding(min_periods=1).median()
        ind["stressed"] = (drawdown < dd_hist_median).astype(float)

        # Drawdown curvature: second difference detects deceleration of losses
        dd_vel = drawdown.diff()
        dd_curv = dd_vel.diff()

        # Smooth curvature to filter single-bar noise (SNR gate)
        dd_curv_smooth = dd_curv.rolling(params.smooth, min_periods=params.smooth).mean()
        ind["curv_positive"] = (dd_curv_smooth > 0).astype(float)

        return ind

    @staticmethod
    def generate_signals(
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: Params,
    ) -> SignalFrame:
        df = pd.DataFrame(index=data.index)

        # Entry: stressed regime AND drawdown curvature turning positive (losses decelerating)
        # Signal-reversal exit is implicit: signal stays 1 while both conditions hold,
        # drops to 0 the bar after either condition flips — no path-dependence needed
        entry = (indicators["stressed"] == 1) & (indicators["curv_positive"] == 1)
        raw = entry.astype(int)

        # Mandatory one-bar shift: signal decided on bar N, filled on bar N+1
        df["signal"] = raw.shift(1).fillna(0).astype(int)
        df["size"] = 1.0

        return SignalFrame(data=df, signal_column="signal", size_column="size")
