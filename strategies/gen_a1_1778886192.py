from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class GapRejectionParams:
    gap_lookback: int = 60
    gap_k: float = 0.75
    atr_period: int = 14
    conv_period: int = 20
    conv_clip: float = 3.0
    stop_k: float = 2.0
    hold_days: int = 4
    size_min: float = 0.3
    size_max: float = 1.0


class GeneratedStrategy(BaseStrategy[GapRejectionParams]):
    """Gap-rejection continuation.

    A bar 'rejects' its overnight gap when the gap and the same-day
    close-to-close return have opposite signs. The strategy then trades in
    the direction of the close-to-close return (the side the cash session
    endorsed), holds 3-5 days, and exits on a fixed ATR volatility stop
    measured from the entry price. Position size scales with the conviction
    of the move (the close-to-close return's rolling z-score).
    """

    strategy_id = "gen_a1_1778886192"

    @classmethod
    def params_type(cls) -> type[GapRejectionParams]:
        return GapRejectionParams

    def warmup_bars(self, params: GapRejectionParams) -> int:
        return int(max(params.gap_lookback, params.atr_period, params.conv_period)) + 1

    def indicators(self, data: pd.DataFrame, params: GapRejectionParams) -> pd.DataFrame:
        close = data["close"]
        open_ = data["open"]
        high = data["high"]
        low = data["low"]
        prev_close = close.shift(1)

        gap = open_ / prev_close - 1.0
        ret = close.pct_change()

        tr = pd.concat(
            [
                high - low,
                (high - prev_close).abs(),
                (low - prev_close).abs(),
            ],
            axis=1,
        ).max(axis=1)
        atr = tr.rolling(params.atr_period, min_periods=params.atr_period).mean()

        gap_std = gap.rolling(params.gap_lookback, min_periods=params.gap_lookback).std()
        gap_thr = gap_std * float(params.gap_k)

        ret_std = ret.rolling(params.conv_period, min_periods=params.conv_period).std()
        ret_std = ret_std.replace(0.0, np.nan)
        conviction = (ret.abs() / ret_std).clip(upper=float(params.conv_clip))

        # A rejected down-gap that closes green -> go long.
        # A rejected up-gap that closes red   -> go short.
        long_entry = (gap < -gap_thr) & (ret > 0.0)
        short_entry = (gap > gap_thr) & (ret < 0.0)

        out = pd.DataFrame(index=data.index)
        out["gap"] = gap
        out["ret"] = ret
        out["atr"] = atr
        out["gap_thr"] = gap_thr
        out["conviction"] = conviction
        out["long_entry"] = long_entry.fillna(False)
        out["short_entry"] = short_entry.fillna(False)
        return out

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: GapRejectionParams,
    ) -> SignalFrame:
        n = len(data)
        close = data["close"].to_numpy(dtype=float)
        atr = indicators["atr"].to_numpy(dtype=float)
        conv = indicators["conviction"].to_numpy(dtype=float)
        long_e = indicators["long_entry"].to_numpy(dtype=bool)
        short_e = indicators["short_entry"].to_numpy(dtype=bool)

        size_min = float(params.size_min)
        size_max = float(params.size_max)
        conv_clip = float(params.conv_clip)
        stop_k = float(params.stop_k)
        hold_days = int(params.hold_days)
        if conv_clip <= 0.0:
            conv_clip = 1.0
        if size_max < size_min:
            size_max = size_min

        sig = np.zeros(n, dtype=int)
        size = np.full(n, size_min, dtype=float)

        def conv_to_size(c: float) -> float:
            if not np.isfinite(c) or c <= 0.0:
                return size_min
            frac = c / conv_clip
            if frac < 0.0:
                frac = 0.0
            elif frac > 1.0:
                frac = 1.0
            return size_min + (size_max - size_min) * frac

        position = 0
        entry_price = 0.0
        entry_atr = 0.0
        cur_size = size_min
        bars_held = 0

        for t in range(n):
            if position == 0:
                a = atr[t]
                if np.isfinite(a) and a > 0.0:
                    if long_e[t]:
                        position = 1
                        entry_price = close[t]
                        entry_atr = a
                        cur_size = conv_to_size(conv[t])
                        bars_held = 0
                        sig[t] = 1
                        size[t] = cur_size
                    elif short_e[t]:
                        position = -1
                        entry_price = close[t]
                        entry_atr = a
                        cur_size = conv_to_size(conv[t])
                        bars_held = 0
                        sig[t] = -1
                        size[t] = cur_size
            else:
                bars_held += 1
                exit_now = False
                if position == 1:
                    if close[t] <= entry_price - stop_k * entry_atr:
                        exit_now = True
                else:
                    if close[t] >= entry_price + stop_k * entry_atr:
                        exit_now = True
                if bars_held >= hold_days:
                    exit_now = True
                if exit_now:
                    sig[t] = 0
                    size[t] = size_min
                    position = 0
                    cur_size = size_min
                else:
                    sig[t] = position
                    size[t] = cur_size

        df = pd.DataFrame(index=data.index)
        df["signal"] = sig
        df["size"] = size
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = df["size"].shift(1).fillna(size_min)
        df["size"] = df["size"].clip(lower=1e-6)
        return SignalFrame(data=df, signal_column="signal", size_column="size")
