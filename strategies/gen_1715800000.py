from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class GeneratedParams:
    """Parameters for the range-compression / expansion strategy.

    range_period:       lookback for the true-range moving average.
    percentile_window:  window over which the current range-average is ranked.
    entry_percentile:   enter long when the range-average rank is at or below
                        this percentile (0-100). Lower = more compressed.
    exit_percentile:    exit when the range-average rank rises at or above this
                        percentile (range has re-expanded / "plastic" break).
    max_hold:           hard exit after this many bars regardless of range.
    size:               position size (fraction of equity, via percent_equity).
    """

    range_period: int = 10
    percentile_window: int = 60
    entry_percentile: float = 20.0
    exit_percentile: float = 70.0
    max_hold: int = 5
    size: float = 1.0


class GeneratedStrategy(BaseStrategy[GeneratedParams]):
    """
    Purpose:
        Long-only volatility-compression strategy. Buys SPY when its average
        true range contracts into a low percentile of its recent distribution
        (an "elastic", coiled state) and exits on either range re-expansion
        (a "plastic" break) or a fixed maximum holding period.

    Inputs:
        OHLCV dataframe with datetime index and lowercase columns:
        open, high, low, close, volume.

    Outputs:
        SignalFrame with `signal` (0/1, long-only) and `size` columns.

    Side effects:
        None.
    """

    strategy_id = "gen_1715800000"
    version = "1.0"
    asset_type = "stock"
    timeframe = "1d"

    @classmethod
    def params_type(cls):
        return GeneratedParams

    def warmup_bars(self, params: GeneratedParams) -> int:
        # true range uses prev close (1 extra bar), then a rolling mean of
        # range_period, then a rolling percentile rank of percentile_window.
        # Longest chain: 1 + range_period + percentile_window.
        return 1 + params.range_period + params.percentile_window

    def indicators(self, data: pd.DataFrame, params: GeneratedParams) -> pd.DataFrame:
        out = pd.DataFrame(index=data.index)

        high = data["high"]
        low = data["low"]
        prev_close = data["close"].shift(1)

        # True range: max of (H-L), |H-prevC|, |L-prevC|
        tr = pd.concat(
            [
                (high - low),
                (high - prev_close).abs(),
                (low - prev_close).abs(),
            ],
            axis=1,
        ).max(axis=1)

        # Smoothed range.
        range_avg = tr.rolling(params.range_period).mean()

        # Rank the current range_avg within its own trailing distribution.
        # rank(pct=True) gives 0..1; *100 -> percentile. The last value in each
        # window is the current bar's rank vs the window.
        range_pctile = range_avg.rolling(params.percentile_window).apply(
            lambda w: pd.Series(w).rank(pct=True).iloc[-1] * 100.0,
            raw=False,
        )

        out["true_range"] = tr
        out["range_avg"] = range_avg
        out["range_pctile"] = range_pctile
        return out

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: GeneratedParams,
    ) -> SignalFrame:
        pctile = indicators["range_pctile"]

        n = len(data)
        raw_signal = np.zeros(n, dtype=int)

        in_position = False
        bars_held = 0

        # State machine over bars. raw_signal here is the *intended* state at
        # bar i's close; it gets shifted by one bar at the end so the fill
        # happens on bar i+1's open.
        for i in range(n):
            p = pctile.iloc[i]

            if not in_position:
                # Enter long only on a valid, sufficiently-compressed reading.
                if not np.isnan(p) and p <= params.entry_percentile:
                    in_position = True
                    bars_held = 0
                    raw_signal[i] = 1
                else:
                    raw_signal[i] = 0
            else:
                bars_held += 1
                # Exit on re-expansion (plastic break) or hard max-hold cap.
                expanded = (not np.isnan(p)) and (p >= params.exit_percentile)
                timed_out = bars_held >= params.max_hold
                if expanded or timed_out:
                    in_position = False
                    bars_held = 0
                    raw_signal[i] = 0
                else:
                    raw_signal[i] = 1

        df = pd.DataFrame(index=data.index)
        df["signal"] = raw_signal
        # MANDATORY one-bar shift: decide on bar N close, fill on bar N+1 open.
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = params.size

        return SignalFrame(data=df, signal_column="signal", size_column="size")
