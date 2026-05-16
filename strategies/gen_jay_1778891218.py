from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class TidalPhaseLockParams:
    osc_lookback: int = 20
    osc_expand_window: int = 10
    midpoint_window: int = 20
    min_hightide_streak: int = 3
    mom_window: int = 5
    mom_lag: int = 5
    min_accel_streak: int = 2
    atr_window: int = 14
    trail_atr_mult: float = 2.0
    base_size: float = 0.95


class GeneratedStrategy(BaseStrategy[TidalPhaseLockParams]):
    strategy_id = "gen_jay_1778891218"

    @classmethod
    def params_type(cls):
        return TidalPhaseLockParams

    @staticmethod
    def warmup_bars(params: TidalPhaseLockParams) -> int:
        return (
            max(
                params.osc_lookback + params.osc_expand_window * 2,
                params.midpoint_window + params.min_hightide_streak + 1,
                params.mom_window + params.mom_lag + params.min_accel_streak + 1,
                params.atr_window,
            )
            + 10
        )

    @staticmethod
    def indicators(data: pd.DataFrame, params: TidalPhaseLockParams) -> pd.DataFrame:
        ind = pd.DataFrame(index=data.index)
        close = data["close"]
        high = data["high"]
        low = data["low"]

        tr = pd.concat(
            [
                high - low,
                (high - close.shift(1)).abs(),
                (low - close.shift(1)).abs(),
            ],
            axis=1,
        ).max(axis=1)
        ind["atr"] = tr.ewm(span=params.atr_window, min_periods=params.atr_window).mean()

        node = close.rolling(params.osc_lookback, min_periods=params.osc_lookback).median()
        osc_amp = (
            (close - node)
            .abs()
            .rolling(params.osc_expand_window, min_periods=params.osc_expand_window)
            .mean()
        )
        osc_amp_prev = osc_amp.shift(params.osc_expand_window)
        ind["regime_expanding"] = (osc_amp > osc_amp_prev).astype(int)

        hl_high = high.rolling(params.midpoint_window, min_periods=params.midpoint_window).max()
        hl_low = low.rolling(params.midpoint_window, min_periods=params.midpoint_window).min()
        midpoint = (hl_high + hl_low) / 2.0
        above_mid = (close > midpoint).astype(int)
        grp_a = (above_mid != above_mid.shift(1)).cumsum()
        ind["streak_hightide"] = (
            (above_mid.groupby(grp_a).cumcount() + 1).where(above_mid == 1, 0)
        )

        mom = close.pct_change(params.mom_window)
        accel = (mom > mom.shift(params.mom_lag)).astype(int)
        grp_b = (accel != accel.shift(1)).cumsum()
        ind["streak_accel"] = (
            (accel.groupby(grp_b).cumcount() + 1).where(accel == 1, 0)
        )

        return ind

    @staticmethod
    def generate_signals(
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: TidalPhaseLockParams,
    ) -> SignalFrame:
        close = data["close"].values
        atr = indicators["atr"].values
        regime = indicators["regime_expanding"].values
        streak_a = indicators["streak_hightide"].values
        streak_b = indicators["streak_accel"].values

        n = len(close)
        raw_signal = np.zeros(n, dtype=np.int64)

        in_trade = False
        hwm = 0.0

        for i in range(n):
            if np.isnan(atr[i]):
                in_trade = False
                hwm = 0.0
                continue

            if in_trade:
                if close[i] > hwm:
                    hwm = close[i]
                stop_level = hwm - params.trail_atr_mult * atr[i]
                if close[i] <= stop_level:
                    raw_signal[i] = 0
                    in_trade = False
                    hwm = 0.0
                else:
                    raw_signal[i] = 1
            else:
                entry = (
                    regime[i] == 1
                    and streak_a[i] >= params.min_hightide_streak
                    and streak_b[i] >= params.min_accel_streak
                )
                if entry:
                    raw_signal[i] = 1
                    in_trade = True
                    hwm = close[i]

        df = pd.DataFrame(index=data.index)
        df["signal"] = pd.Series(raw_signal, index=data.index).shift(1).fillna(0).astype(int)
        df["size"] = params.base_size

        return SignalFrame(data=df, signal_column="signal", size_column="size")
