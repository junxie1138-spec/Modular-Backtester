from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class SeasonalAccelParams:
    roc_len: int = 10
    accel_len: int = 5
    seasonal_threshold: float = 0.0
    spike_window: int = 20
    spike_mult: float = 2.5
    refractory_bars: int = 3
    profit_target_pct: float = 0.04
    max_hold_bars: int = 10


class GeneratedStrategy(BaseStrategy[SeasonalAccelParams]):
    strategy_id = "gen_a1_1778884705"

    @classmethod
    def params_type(cls):
        return SeasonalAccelParams

    def warmup_bars(self, params: SeasonalAccelParams) -> int:
        roc_chain = int(params.roc_len) + int(params.accel_len) + 1
        spike_chain = int(params.spike_window) + 1
        refractory_chain = int(params.refractory_bars) + 1
        return int(max(roc_chain, spike_chain, refractory_chain, 2))

    def indicators(self, data: pd.DataFrame, params: SeasonalAccelParams) -> pd.DataFrame:
        close = data["close"]
        ret = close.pct_change()

        # Primitive 1: endogenous month-of-year seasonal bias.
        # Expanding mean of same-month daily returns, lagged one bar so the
        # current bar never contributes to its own seasonal score.
        month = pd.Series(data.index.month, index=data.index)
        seasonal_mean = ret.groupby(month).transform(
            lambda s: s.shift(1).expanding(min_periods=1).mean()
        )
        seasonal_favorable = (seasonal_mean > params.seasonal_threshold).fillna(False)

        # Primitive 2: rate-of-change acceleration (positive second derivative).
        roc = close.pct_change(int(params.roc_len))
        accel = roc.diff(int(params.accel_len))
        accel_positive = (accel > 0.0).fillna(False)

        # Refractory window: suppress entries on and shortly after a return spike.
        vol = ret.rolling(int(params.spike_window), min_periods=int(params.spike_window)).std()
        spike = (ret.abs() > (params.spike_mult * vol)).fillna(False)
        in_refractory = (
            spike.astype(float)
            .rolling(int(params.refractory_bars) + 1, min_periods=1)
            .max()
            > 0.0
        ).fillna(False)

        out = pd.DataFrame(index=data.index)
        out["seasonal_mean"] = seasonal_mean
        out["seasonal_favorable"] = seasonal_favorable
        out["roc"] = roc
        out["accel"] = accel
        out["accel_positive"] = accel_positive
        out["in_refractory"] = in_refractory
        return out

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: SeasonalAccelParams,
    ) -> SignalFrame:
        close = data["close"].to_numpy(dtype=float)
        n = len(close)

        favorable = indicators["seasonal_favorable"].to_numpy()
        accel_pos = indicators["accel_positive"].to_numpy()
        refractory = indicators["in_refractory"].to_numpy()

        raw = np.zeros(n, dtype=np.int64)
        in_pos = False
        entry_price = 0.0
        entry_bar = 0

        target = float(params.profit_target_pct)
        max_hold = int(params.max_hold_bars)

        for i in range(n):
            if not in_pos:
                # Two-primitive AND: both must agree, and not in refractory.
                if (
                    bool(favorable[i])
                    and bool(accel_pos[i])
                    and not bool(refractory[i])
                ):
                    in_pos = True
                    entry_price = close[i]
                    entry_bar = i
                    raw[i] = 1
            else:
                held = i - entry_bar
                gain = (close[i] / entry_price - 1.0) if entry_price > 0.0 else 0.0
                # Exit: profit-target OR time-stop, whichever fires first.
                if gain >= target or held >= max_hold:
                    in_pos = False
                    raw[i] = 0
                else:
                    raw[i] = 1

        df = pd.DataFrame(index=data.index)
        df["signal"] = raw
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = 1.0
        return SignalFrame(data=df, signal_column="signal", size_column="size")
