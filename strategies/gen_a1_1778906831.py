from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class GeneratedParams:
    rank_window: int = 20
    rank_threshold: float = 0.55
    roc_lag: int = 5
    accel_lag: int = 3
    roc_smooth: int = 3
    min_accel: float = 0.0


class GeneratedStrategy(BaseStrategy[GeneratedParams]):
    strategy_id = "gen_a1_1778906831"

    @classmethod
    def params_type(cls) -> type[GeneratedParams]:
        return GeneratedParams

    @staticmethod
    def warmup_bars(params: GeneratedParams) -> int:
        rw = max(int(params.rank_window), 1)
        momentum_span = (
            max(int(params.roc_lag), 1)
            + max(int(params.roc_smooth), 1)
            + max(int(params.accel_lag), 1)
        )
        return int(max(rw, momentum_span)) + 5

    @staticmethod
    def indicators(data: pd.DataFrame, params: GeneratedParams) -> pd.DataFrame:
        close = data["close"].astype(float)
        high = data["high"].astype(float)
        low = data["low"].astype(float)

        rw = max(int(params.rank_window), 2)
        roc_lag = max(int(params.roc_lag), 1)
        accel_lag = max(int(params.accel_lag), 1)
        roc_smooth = max(int(params.roc_smooth), 1)

        out = pd.DataFrame(index=data.index)

        # --- Primitive 1: relative position within the rolling high-low range ---
        roll_max = high.rolling(rw, min_periods=rw).max()
        roll_min = low.rolling(rw, min_periods=rw).min()
        span = (roll_max - roll_min)
        span = span.where(span > 0.0, np.nan)
        relpos = (close - roll_min) / span
        out["relpos"] = relpos.clip(lower=0.0, upper=1.0)

        # --- Primitive 2: rate-of-change acceleration (smoothed second difference) ---
        roc = close.pct_change(roc_lag)
        roc_s = roc.ewm(span=roc_smooth, adjust=False).mean()
        # acceleration = change in the rate of change over accel_lag bars
        accel = roc_s.diff(accel_lag)
        out["roc"] = roc_s
        out["accel"] = accel

        return out

    @staticmethod
    def generate_signals(
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: GeneratedParams,
    ) -> SignalFrame:
        relpos = indicators["relpos"]
        accel = indicators["accel"]

        # Two-primitive AND: price sits in the upper part of its range
        # AND momentum is accelerating. Comparisons against NaN yield False,
        # so warmup bars are naturally flat.
        pos_ok = (relpos > float(params.rank_threshold)).fillna(False)
        accel_ok = (accel > float(params.min_accel)).fillna(False)
        entry_condition = pos_ok & accel_ok

        # Long-only. Hold while the conjunction holds; the signal-reversal exit
        # is the conjunction breaking - signal falls to 0 the moment either
        # primitive flips. No separate stop logic.
        raw_signal = entry_condition.astype(int)

        # Conviction-scaled size from the relative-position primitive.
        size = (0.5 + 0.5 * relpos.clip(lower=0.0, upper=1.0)).fillna(0.5)
        size = size.clip(lower=0.5, upper=1.0)

        df = pd.DataFrame(index=data.index)
        df["signal"] = raw_signal.astype(int)
        df["size"] = size.astype(float)

        # MANDATORY one-bar shift: decide on bar N close, fill on bar N+1.
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = df["size"].shift(1).fillna(1.0).astype(float)

        return SignalFrame(data=df, signal_column="signal", size_column="size")
