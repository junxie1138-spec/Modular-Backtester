from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class GeneratedParams:
    roc_period: int = 10
    accel_smooth: int = 3
    lookback: int = 60
    q: float = 0.85
    atr_period: int = 14
    breakeven_pct: float = 0.04
    init_stop_mult: float = 2.0
    trail_mult: float = 3.0
    refractory: int = 5


class GeneratedStrategy(BaseStrategy[GeneratedParams]):
    strategy_id = "gen_a1_1778913466"

    @classmethod
    def params_type(cls) -> type[GeneratedParams]:
        return GeneratedParams

    @staticmethod
    def warmup_bars(params: GeneratedParams) -> int:
        roc_chain = params.roc_period + 1 + params.accel_smooth
        pctl_chain = params.lookback + 1
        atr_chain = params.atr_period + 1
        return int(max(roc_chain, pctl_chain, atr_chain)) + 5

    @staticmethod
    def indicators(data: pd.DataFrame, params: GeneratedParams) -> pd.DataFrame:
        close = data["close"]
        high = data["high"]
        low = data["low"]

        ind = pd.DataFrame(index=data.index)

        # Rate-of-change and its (smoothed) acceleration - the second difference.
        roc = close.pct_change(params.roc_period)
        roc_accel = roc.diff()
        accel_sm = roc_accel.rolling(params.accel_smooth, min_periods=params.accel_smooth).mean()
        ind["accel"] = accel_sm

        # Percentile breakout level: q-th percentile of the TRAILING close
        # distribution (prior closes only, current close excluded).
        pctl_level = (
            close.shift(1)
            .rolling(params.lookback, min_periods=params.lookback)
            .quantile(params.q)
        )
        ind["pctl_level"] = pctl_level

        # Average True Range (Wilder-style rolling mean of true range).
        prev_close = close.shift(1)
        tr = pd.concat(
            [
                (high - low),
                (high - prev_close).abs(),
                (low - prev_close).abs(),
            ],
            axis=1,
        ).max(axis=1)
        atr = tr.rolling(params.atr_period, min_periods=params.atr_period).mean()
        ind["atr"] = atr

        ind["breakout"] = (close > pctl_level).astype(float)

        return ind

    @staticmethod
    def generate_signals(
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: GeneratedParams,
    ) -> SignalFrame:
        close = data["close"].to_numpy(dtype=float)
        high = data["high"].to_numpy(dtype=float)
        low = data["low"].to_numpy(dtype=float)

        accel = indicators["accel"].to_numpy(dtype=float)
        pctl_level = indicators["pctl_level"].to_numpy(dtype=float)
        atr = indicators["atr"].to_numpy(dtype=float)
        breakout = indicators["breakout"].to_numpy(dtype=float)

        n = len(data)
        raw = np.zeros(n, dtype=int)

        in_pos = False
        entry_price = 0.0
        stop = 0.0
        be_armed = False
        cooldown_until = 0

        for i in range(n):
            a = atr[i]
            c = close[i]
            h = high[i]
            l = low[i]

            if in_pos:
                # Path-dependent stop management: breakeven-then-trail.
                if not np.isnan(a):
                    if not be_armed and h >= entry_price * (1.0 + params.breakeven_pct):
                        stop = max(stop, entry_price)
                        be_armed = True
                    if be_armed:
                        stop = max(stop, c - params.trail_mult * a)

                if l <= stop:
                    # Stop hit - go flat this bar, start refractory lockout.
                    in_pos = False
                    be_armed = False
                    raw[i] = 0
                    cooldown_until = i + params.refractory
                else:
                    raw[i] = 1
            else:
                valid = (
                    not np.isnan(accel[i])
                    and not np.isnan(pctl_level[i])
                    and not np.isnan(atr[i])
                    and not np.isnan(breakout[i])
                )
                if (
                    valid
                    and i >= cooldown_until
                    and breakout[i] > 0.5
                    and accel[i] > 0.0
                ):
                    in_pos = True
                    entry_price = c
                    stop = c - params.init_stop_mult * atr[i]
                    be_armed = False
                    raw[i] = 1
                else:
                    raw[i] = 0

        df = pd.DataFrame(index=data.index)
        df["signal"] = raw
        df["size"] = 1.0
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)

        return SignalFrame(data=df, signal_column="signal", size_column="size")
