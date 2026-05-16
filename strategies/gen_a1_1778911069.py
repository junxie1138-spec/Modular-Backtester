from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class CompressionSpringParams:
    roc_period: int = 5
    smooth_period: int = 5
    comp_short: int = 5
    comp_long: int = 40
    comp_threshold: float = 0.85
    tension_gain: float = 1.5
    base_size: float = 1.0


class GeneratedStrategy(BaseStrategy[CompressionSpringParams]):
    """Range-compression spring: enter on confirmed positive ROC-acceleration
    while the recent range is compressed; exit only when acceleration flips."""

    strategy_id = "gen_a1_1778911069"

    @classmethod
    def params_type(cls) -> type[CompressionSpringParams]:
        return CompressionSpringParams

    @staticmethod
    def warmup_bars(params: CompressionSpringParams) -> int:
        # roc uses pct_change(roc_period), then a rolling mean of smooth_period,
        # then a .diff() -> roc_period + smooth_period + 1 bars minimum.
        roc_chain = params.roc_period + params.smooth_period + 2
        return int(max(roc_chain, params.comp_long) + 5)

    @staticmethod
    def indicators(data: pd.DataFrame, params: CompressionSpringParams) -> pd.DataFrame:
        close = data["close"]
        high = data["high"]
        low = data["low"]

        # Rate-of-change acceleration: smoothed ROC, then its bar-to-bar change.
        roc = close.pct_change(params.roc_period)
        roc_smooth = roc.rolling(
            params.smooth_period, min_periods=params.smooth_period
        ).mean()
        accel = roc_smooth.diff()

        # Range compression: short-window mean range vs long-window baseline.
        bar_range = (high - low).abs()
        short_range = bar_range.rolling(
            params.comp_short, min_periods=params.comp_short
        ).mean()
        long_range = bar_range.rolling(
            params.comp_long, min_periods=params.comp_long
        ).mean()
        compression = short_range / long_range.replace(0.0, np.nan)

        ind = pd.DataFrame(index=data.index)
        ind["accel"] = accel
        ind["compression"] = compression
        return ind

    @staticmethod
    def generate_signals(
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: CompressionSpringParams,
    ) -> SignalFrame:
        accel = indicators["accel"]
        compression = indicators["compression"]

        accel_pos = (accel > 0.0).fillna(False).to_numpy()
        accel_neg = (accel < 0.0).fillna(False).to_numpy()
        compressed = (compression < params.comp_threshold).fillna(False).to_numpy()
        valid = (accel.notna() & compression.notna()).to_numpy()

        n = len(data)
        entry = np.zeros(n, dtype=bool)
        exit_cond = np.zeros(n, dtype=bool)

        # Two-bar confirmation: acceleration must hold its sign across bar i-1
        # and bar i. Entry also requires the range to be compressed now.
        for i in range(1, n):
            if valid[i] and valid[i - 1]:
                entry[i] = accel_pos[i] and accel_pos[i - 1] and compressed[i]
                exit_cond[i] = accel_neg[i] and accel_neg[i - 1]

        # Signal-reversal exit: hold until the entry's acceleration sign flips.
        raw = np.zeros(n, dtype=np.int64)
        position = 0
        for i in range(n):
            if position == 0:
                if entry[i]:
                    position = 1
            else:
                if exit_cond[i]:
                    position = 0
            raw[i] = position

        # Spring tension: tighter compression -> larger size (elastic stiffness).
        comp_clipped = compression.clip(lower=0.0, upper=params.comp_threshold)
        tension = (params.comp_threshold - comp_clipped) / params.comp_threshold
        size = params.base_size * (1.0 + params.tension_gain * tension)
        size = size.fillna(params.base_size).clip(lower=0.01)

        df = pd.DataFrame(index=data.index)
        df["signal"] = pd.Series(raw, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = size.to_numpy()
        return SignalFrame(data=df, signal_column="signal", size_column="size")
