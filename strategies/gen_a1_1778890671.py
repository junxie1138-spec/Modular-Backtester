from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class TailQuietParams:
    ret_lookback: int = 20
    pct_window: int = 252
    low_pct: float = 0.15
    high_pct: float = 0.85
    tail_count_enter: int = 0
    tail_count_loose: int = 3
    hold_bars: int = 7
    trend_ma: int = 200
    size_min: float = 0.6
    size_max: float = 1.0


class GeneratedStrategy(BaseStrategy[TailQuietParams]):
    """Range-compression long-only strategy.

    Compression is measured purely on close-to-close returns: count how many
    of the last `ret_lookback` returns crossed outside the empirical
    [low_pct, high_pct] quantile band of a long return history. When that
    tail-touch count collapses to `tail_count_enter` (the return distribution
    has stopped reaching its own tails) the strategy goes long for exactly
    `hold_bars` bars. An armed/disarmed hysteresis latch requires the count
    to first climb back above `tail_count_loose` before a new entry can arm.
    """

    strategy_id = "gen_a1_1778890671"

    @classmethod
    def params_type(cls):
        return TailQuietParams

    @staticmethod
    def warmup_bars(params: TailQuietParams) -> int:
        return int(max(params.pct_window + params.ret_lookback + 2,
                       params.trend_ma + 2))

    @staticmethod
    def indicators(data: pd.DataFrame, params: TailQuietParams) -> pd.DataFrame:
        close = data["close"].astype(float)
        ret = close.pct_change()

        pw = max(int(params.pct_window), 2)
        rl = max(int(params.ret_lookback), 1)
        lp = min(max(float(params.low_pct), 0.0), 1.0)
        hp = min(max(float(params.high_pct), 0.0), 1.0)
        if hp < lp:
            lp, hp = hp, lp

        # Empirical percentile band of the long-run return distribution.
        low_band = ret.rolling(pw, min_periods=pw).quantile(lp)
        high_band = ret.rolling(pw, min_periods=pw).quantile(hp)

        # A bar 'touches a tail' if its return falls outside the band.
        tail_raw = (ret < low_band) | (ret > high_band)
        tail_touch = tail_raw.astype(float)
        invalid = ret.isna() | low_band.isna() | high_band.isna()
        tail_touch = tail_touch.where(~invalid, other=np.nan)

        # Number of tail touches in the recent window (NaN during warmup).
        tail_count = tail_touch.rolling(rl, min_periods=rl).sum()

        tm = max(int(params.trend_ma), 1)
        ma = close.rolling(tm, min_periods=tm).mean()
        regime = (close > ma).fillna(False)

        out = pd.DataFrame(index=data.index)
        out["ret"] = ret
        out["low_band"] = low_band
        out["high_band"] = high_band
        out["tail_count"] = tail_count
        out["regime"] = regime.astype(float)
        return out

    @staticmethod
    def generate_signals(data: pd.DataFrame, indicators: pd.DataFrame,
                         ctx: StrategyContext,
                         params: TailQuietParams) -> SignalFrame:
        n = len(data)
        tail_count = indicators["tail_count"].to_numpy(dtype=float)
        regime = indicators["regime"].to_numpy(dtype=float)

        hold = max(int(params.hold_bars), 1)
        enter_thr = int(params.tail_count_enter)
        loose_thr = int(params.tail_count_loose)
        s_min = float(params.size_min)
        s_max = float(params.size_max)
        if s_max < s_min:
            s_min, s_max = s_max, s_min
        denom = float(max(loose_thr, 1))

        sig = np.zeros(n, dtype=int)
        size = np.ones(n, dtype=float)

        armed = False
        in_pos = False
        entry_idx = -1
        entry_size = 1.0

        for i in range(n):
            tc = tail_count[i]
            if np.isnan(tc):
                continue

            # Hysteresis: re-arm only after the count climbs back to 'loose'.
            if tc >= loose_thr:
                armed = True

            # Fixed-bar exit: flat exactly `hold` bars after entry.
            if in_pos:
                if i - entry_idx >= hold:
                    in_pos = False
                else:
                    sig[i] = 1
                    size[i] = entry_size

            if not in_pos:
                if armed and tc <= enter_thr and regime[i] > 0.5:
                    strength = (denom - tc) / denom
                    if strength < 0.0:
                        strength = 0.0
                    elif strength > 1.0:
                        strength = 1.0
                    entry_size = s_min + (s_max - s_min) * strength
                    in_pos = True
                    entry_idx = i
                    armed = False
                    sig[i] = 1
                    size[i] = entry_size

        df = pd.DataFrame(index=data.index)
        df["signal"] = (
            pd.Series(sig, index=data.index).shift(1).fillna(0).astype(int)
        )
        sz = pd.Series(size, index=data.index).shift(1).fillna(1.0)
        df["size"] = sz.clip(lower=1e-6)
        return SignalFrame(data=df, signal_column="signal", size_column="size")
