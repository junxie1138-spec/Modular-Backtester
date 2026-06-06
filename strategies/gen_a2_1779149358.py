from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class Params:
    rank_window: int = 60
    short_window: int = 10
    range_sum_bars: int = 5
    atr_window: int = 14
    close_rank_hi: float = 0.80
    range_rank_lo: float = 0.20
    breakeven_pct: float = 0.03
    init_stop_atr: float = 3.0
    trail_atr_mult: float = 2.5
    max_hold: int = 5


class GeneratedStrategy(BaseStrategy[Params]):
    strategy_id = "gen_a2_1779149358"

    @classmethod
    def params_type(cls):
        return Params

    @staticmethod
    def warmup_bars(params: Params) -> int:
        return int(
            max(
                params.rank_window + params.range_sum_bars,
                params.short_window,
                params.atr_window,
            )
            + 2
        )

    @staticmethod
    def indicators(data: pd.DataFrame, params: Params) -> pd.DataFrame:
        high = data["high"]
        low = data["low"]
        close = data["close"]
        prev_close = close.shift(1)

        tr = pd.concat(
            [
                high - low,
                (high - prev_close).abs(),
                (low - prev_close).abs(),
            ],
            axis=1,
        ).max(axis=1)

        atr = tr.rolling(params.atr_window, min_periods=params.atr_window).mean()

        close_rank = close.rolling(
            params.short_window, min_periods=params.short_window
        ).rank(pct=True)

        range_sum = tr.rolling(
            params.range_sum_bars, min_periods=params.range_sum_bars
        ).sum()
        range_rank = range_sum.rolling(
            params.rank_window, min_periods=params.rank_window
        ).rank(pct=True)

        out = pd.DataFrame(index=data.index)
        out["atr"] = atr
        out["close_rank"] = close_rank
        out["range_rank"] = range_rank
        return out

    @staticmethod
    def generate_signals(data, indicators, ctx, params):
        n = len(data)
        high = data["high"].to_numpy(dtype=float)
        low = data["low"].to_numpy(dtype=float)
        close = data["close"].to_numpy(dtype=float)
        atr = indicators["atr"].to_numpy(dtype=float)
        close_rank = indicators["close_rank"].to_numpy(dtype=float)
        range_rank = indicators["range_rank"].to_numpy(dtype=float)

        # Two-primitive AND: both rolling-rank conditions must agree.
        # NaN comparisons evaluate False, so warmup bars never trigger entry.
        entry_cond = (close_rank >= params.close_rank_hi) & (
            range_rank <= params.range_rank_lo
        )

        signal = np.zeros(n, dtype=int)
        warmup = GeneratedStrategy.warmup_bars(params)

        in_pos = False
        entry_price = 0.0
        entry_idx = 0
        stop = 0.0
        be_armed = False
        run_high = 0.0

        for i in range(n):
            if not in_pos:
                if (
                    i >= warmup
                    and bool(entry_cond[i])
                    and np.isfinite(atr[i])
                    and atr[i] > 0.0
                ):
                    in_pos = True
                    entry_price = close[i]
                    entry_idx = i
                    stop = entry_price - params.init_stop_atr * atr[i]
                    be_armed = False
                    run_high = high[i]
                    signal[i] = 1
            else:
                if high[i] > run_high:
                    run_high = high[i]

                # Breakeven: once price reaches +X%, lift the stop to entry.
                if (not be_armed) and run_high >= entry_price * (
                    1.0 + params.breakeven_pct
                ):
                    be_armed = True
                    if entry_price > stop:
                        stop = entry_price

                # Trail: after breakeven, ratchet the stop up by k*ATR off the run high.
                if be_armed and np.isfinite(atr[i]):
                    trail = run_high - params.trail_atr_mult * atr[i]
                    if trail > stop:
                        stop = trail

                exit_now = (low[i] <= stop) or (
                    (i - entry_idx) >= params.max_hold
                )
                if exit_now:
                    in_pos = False
                    signal[i] = 0
                else:
                    signal[i] = 1

        df = pd.DataFrame(index=data.index)
        df["signal"] = signal
        df["size"] = 1.0
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")
