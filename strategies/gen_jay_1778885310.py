from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class Params:
    roc_window: int = 10
    accel_window: int = 5
    accel_threshold: float = 0.0
    growth_lag: int = 5
    growth_threshold: float = 1.10


class GeneratedStrategy(BaseStrategy[Params]):
    strategy_id = "gen_jay_1778885310"

    @classmethod
    def params_type(cls):
        return Params

    @staticmethod
    def warmup_bars(params: Params) -> int:
        # roc_window bars for ROC, +1 for diff, +accel_window for rolling mean,
        # +growth_lag for the epidemic ratio shift, +4 safety buffer
        return params.roc_window + params.accel_window + params.growth_lag + 6

    @staticmethod
    def indicators(data: pd.DataFrame, params: Params) -> pd.DataFrame:
        close = data["close"]

        # Primary: percent rate-of-change over roc_window bars
        roc = close.pct_change(params.roc_window)

        # Acceleration: smoothed first difference of ROC (second derivative of price)
        accel = roc.diff().rolling(params.accel_window).mean()

        # Epidemic growth ratio: ROC_t / ROC_{t - growth_lag}
        # Captures whether momentum is compounding (spreading) vs decelerating (recovering)
        # Only valid when lagged ROC is meaningfully positive to avoid division artifacts
        roc_lag = roc.shift(params.growth_lag)
        safe_lag = roc_lag.where(roc_lag > 1e-4, other=np.nan)
        growth_ratio = roc / safe_lag

        ind = pd.DataFrame(index=data.index)
        ind["roc"] = roc
        ind["accel"] = accel
        ind["growth_ratio"] = growth_ratio
        return ind

    @staticmethod
    def generate_signals(
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: Params,
    ) -> SignalFrame:
        ind = indicators

        # Primitive 1: acceleration is positive (momentum is speeding up)
        p1 = (ind["accel"] > params.accel_threshold) & ind["accel"].notna()

        # Primitive 2: epidemic growth ratio exceeds threshold AND ROC is positive
        # Ensures momentum is compounding on a positive base, not recovering from negative
        p2 = (
            (ind["growth_ratio"] > params.growth_threshold)
            & (ind["roc"] > 0.0)
            & ind["growth_ratio"].notna()
        )

        # AND gate: both primitives must agree for entry
        entry = (p1 & p2).values

        # Signal-reversal exit: hold while condition holds, exit immediately when it flips
        n = len(entry)
        raw = np.zeros(n, dtype=np.int64)
        in_pos = False
        for i in range(n):
            if in_pos:
                if entry[i]:
                    raw[i] = 1
                else:
                    raw[i] = 0
                    in_pos = False
            else:
                if entry[i]:
                    raw[i] = 1
                    in_pos = True

        # Size: scale by acceleration magnitude — stronger acceleration => larger position
        accel_abs = ind["accel"].abs()
        q99 = accel_abs.quantile(0.99)
        if q99 > 0.0:
            size_raw = (accel_abs / q99).clip(lower=0.5, upper=1.0)
        else:
            size_raw = pd.Series(1.0, index=data.index)
        size_raw = size_raw.fillna(1.0)

        df = pd.DataFrame(index=data.index)
        df["signal"] = raw
        df["size"] = size_raw.values

        # Mandatory 1-bar shift: decision on bar N executes on bar N+1
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = df["size"].shift(1).fillna(1.0)

        return SignalFrame(data=df, signal_column="signal", size_column="size")
