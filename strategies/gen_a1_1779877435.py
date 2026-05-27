from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class QueueOverflowParams:
    queue_capacity: int = 20
    streak_threshold: int = 3
    atr_period: int = 14
    atr_stop_mult: float = 2.5
    max_hold_bars: int = 20
    require_regime_filter: bool = True
    long_ma_period: int = 200


class GeneratedStrategy(BaseStrategy[QueueOverflowParams]):
    strategy_id = "gen_a1_1779877435"

    @classmethod
    def params_type(cls):
        return QueueOverflowParams

    @classmethod
    def warmup_bars(cls, params: QueueOverflowParams) -> int:
        return int(max(
            params.queue_capacity + params.streak_threshold + 2,
            params.atr_period + 2,
            params.long_ma_period + 2,
        ))

    @classmethod
    def indicators(cls, data: pd.DataFrame, params: QueueOverflowParams) -> pd.DataFrame:
        close = data["close"]
        high = data["high"]
        low = data["low"]

        # Fixed-capacity queue of prior closes: rolling max of the last N CLOSED bars (excludes today).
        prior_max_close = close.shift(1).rolling(
            params.queue_capacity, min_periods=params.queue_capacity
        ).max()

        # Overflow event: today's close exceeds the queue's current ceiling.
        overflow = (close > prior_max_close).astype(float)
        overflow = overflow.where(prior_max_close.notna(), 0.0)

        # Consecutive-streak count of overflow events (resets on any miss).
        is_new = overflow.fillna(0.0).astype(int)
        miss = (is_new == 0).astype(int)
        group = miss.cumsum()
        streak = is_new.groupby(group).cumsum().astype(float)
        streak = streak.where(prior_max_close.notna(), np.nan)

        # True Range / ATR (simple mean).
        prev_close = close.shift(1)
        tr = pd.concat([
            (high - low),
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ], axis=1).max(axis=1)
        atr = tr.rolling(params.atr_period, min_periods=params.atr_period).mean()

        long_ma = close.rolling(
            params.long_ma_period, min_periods=params.long_ma_period
        ).mean()

        return pd.DataFrame({
            "prior_max_close": prior_max_close,
            "overflow": overflow,
            "streak": streak,
            "atr": atr,
            "long_ma": long_ma,
        }, index=data.index)

    @classmethod
    def generate_signals(
        cls,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: QueueOverflowParams,
    ) -> SignalFrame:
        close = data["close"].to_numpy(dtype=float)
        streak = indicators["streak"].to_numpy(dtype=float)
        atr = indicators["atr"].to_numpy(dtype=float)
        long_ma = indicators["long_ma"].to_numpy(dtype=float)

        n = len(close)
        raw = np.zeros(n, dtype=np.int64)
        thresh = float(params.streak_threshold)

        in_pos = False
        entry_idx = -1
        hwm = -np.inf

        for i in range(n):
            if not in_pos:
                if i < 1:
                    continue
                s_today = streak[i]
                s_yest = streak[i - 1]
                if np.isnan(s_today) or np.isnan(s_yest):
                    continue
                if np.isnan(atr[i]) or atr[i] <= 0.0:
                    continue
                if params.require_regime_filter:
                    if np.isnan(long_ma[i]) or close[i] <= long_ma[i]:
                        continue
                # Two-bar confirmation: streak sits at/above threshold on TWO consecutive bars.
                if s_today >= thresh and s_yest >= thresh:
                    in_pos = True
                    entry_idx = i
                    hwm = close[i]
                    raw[i] = 1
            else:
                # Ratcheting in-trade high-water mark.
                if close[i] > hwm:
                    hwm = close[i]
                if not np.isnan(atr[i]) and atr[i] > 0.0:
                    stop_level = hwm - params.atr_stop_mult * atr[i]
                else:
                    stop_level = -np.inf
                bars_held = i - entry_idx
                exit_now = (close[i] < stop_level) or (bars_held >= params.max_hold_bars)
                if exit_now:
                    in_pos = False
                    entry_idx = -1
                    hwm = -np.inf
                    raw[i] = 0
                else:
                    raw[i] = 1

        signal = pd.Series(raw, index=data.index, dtype=np.int64)
        signal = signal.shift(1).fillna(0).astype(int)
        size = pd.Series(np.ones(n, dtype=float), index=data.index)

        df = pd.DataFrame({"signal": signal, "size": size}, index=data.index)
        return SignalFrame(data=df, signal_column="signal", size_column="size")
