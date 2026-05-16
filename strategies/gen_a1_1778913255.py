from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class TrendStreakParams:
    ema_len: int = 50
    sma_len: int = 200
    pct_window: int = 252
    pct_thresh: float = 0.80
    profit_target: float = 0.08
    max_hold: int = 20
    refractory: int = 5


class GeneratedStrategy(BaseStrategy[TrendStreakParams]):
    """Long-only trend-strength strategy.

    Streak primitive: consecutive bars whose close sits above a trend EMA.
    This dwell-time streak survives single down bars and resets only on a
    genuine trend break, so it measures regime persistence. Entry fires when
    the streak crosses up through its own rolling-quantile threshold (a
    percentile level, not a fixed number of bars), price is rising, and the
    long-term regime is bullish. Position is closed on a profit target or a
    hard time-stop, whichever comes first, and a refractory window blocks
    immediate re-entry after any exit.
    """

    strategy_id = "gen_a1_1778913255"

    @classmethod
    def params_type(cls):
        return TrendStreakParams

    @staticmethod
    def warmup_bars(params: TrendStreakParams) -> int:
        return int(max(params.ema_len, params.sma_len, params.pct_window)) + 1

    @staticmethod
    def indicators(data: pd.DataFrame, params: TrendStreakParams) -> pd.DataFrame:
        close = data["close"]

        ema = close.ewm(span=int(max(2, params.ema_len)), adjust=False).mean()
        sma = close.rolling(
            int(max(2, params.sma_len)),
            min_periods=int(max(2, params.sma_len)),
        ).mean()

        # Dwell-time streak: consecutive bars closing above the trend EMA.
        above = close > ema
        reset_grp = (~above).cumsum()
        streak = above.astype(int).groupby(reset_grp).cumsum()

        # Percentile threshold instead of a fixed level: the streak must
        # exceed the pct_thresh-quantile of its own trailing distribution.
        w = int(max(2, params.pct_window))
        thr = float(min(0.99, max(0.01, params.pct_thresh)))
        streak_thresh = streak.rolling(w, min_periods=w).quantile(thr)

        rising = close > close.shift(1)
        regime = close > sma
        cross_up = (streak > streak_thresh) & (streak.shift(1) <= streak_thresh)
        entry_raw = (cross_up & rising & regime).fillna(False)

        out = pd.DataFrame(index=data.index)
        out["ema"] = ema
        out["sma"] = sma
        out["streak"] = streak.astype(float)
        out["streak_thresh"] = streak_thresh
        out["entry_raw"] = entry_raw.astype(int)
        return out

    @staticmethod
    def generate_signals(
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: TrendStreakParams,
    ) -> SignalFrame:
        n = len(data)
        close = data["close"].to_numpy(dtype=float)
        entry_raw = indicators["entry_raw"].fillna(0).to_numpy(dtype=int)

        signal = np.zeros(n, dtype=int)
        pt = float(params.profit_target)
        max_hold = int(max(1, params.max_hold))
        refractory = int(max(0, params.refractory))

        in_pos = False
        entry_price = 0.0
        entry_bar = -1
        last_exit_bar = -(10 ** 9)

        # Path-dependent exit: profit-target or time-stop, whichever fires
        # first, plus a refractory lockout after every exit.
        for i in range(n):
            if in_pos:
                bars_held = i - entry_bar
                gain = (close[i] / entry_price - 1.0) if entry_price > 0.0 else 0.0
                if gain >= pt or bars_held >= max_hold:
                    signal[i] = 0
                    in_pos = False
                    last_exit_bar = i
                else:
                    signal[i] = 1
            else:
                if entry_raw[i] == 1 and (i - last_exit_bar) > refractory:
                    in_pos = True
                    entry_price = close[i]
                    entry_bar = i
                    signal[i] = 1

        df = pd.DataFrame(index=data.index)
        df["signal"] = signal
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = 1.0
        return SignalFrame(data=df, signal_column="signal", size_column="size")
