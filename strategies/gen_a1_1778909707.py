from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class CapacityOverflowParams:
    queue_window: int = 10
    capacity_window: int = 120
    capacity_pct: float = 0.55
    mom_lookback: int = 5
    rank_window: int = 60
    entry_rank_thr: float = 0.60
    trend_ma: int = 100
    hold_bars: int = 4
    base_size: float = 1.0


class GeneratedStrategy(BaseStrategy[CapacityOverflowParams]):
    strategy_id = "gen_a1_1778909707"

    @classmethod
    def params_type(cls) -> type[CapacityOverflowParams]:
        return CapacityOverflowParams

    @staticmethod
    def warmup_bars(params: CapacityOverflowParams) -> int:
        return int(
            max(
                params.capacity_window + params.queue_window,
                params.rank_window + params.mom_lookback,
                params.trend_ma,
            )
            + 5
        )

    def indicators(self, data: pd.DataFrame, params: CapacityOverflowParams) -> pd.DataFrame:
        close = data["close"]
        high = data["high"]
        low = data["low"]
        prev_close = close.shift(1)

        # True range -> the per-bar 'arrival' into the range queue.
        tr = pd.concat(
            [
                high - low,
                (high - prev_close).abs(),
                (low - prev_close).abs(),
            ],
            axis=1,
        ).max(axis=1)

        # Fixed-capacity queue: cumulative true range over a rolling window.
        queue_fill = tr.rolling(params.queue_window, min_periods=params.queue_window).sum()

        # Adaptive capacity ceiling: a rolling percentile of the queue's own history.
        capacity = queue_fill.rolling(
            params.capacity_window, min_periods=params.capacity_window
        ).quantile(params.capacity_pct)

        # Calm regime: the queue has spare capacity (no overflow).
        calm = queue_fill <= capacity

        # Momentum primitive expressed as a rolling percentile rank.
        roc = close / close.shift(params.mom_lookback) - 1.0
        mom_rank = roc.rolling(params.rank_window, min_periods=params.rank_window).rank(pct=True)

        # Long-only trend filter.
        trend_ma = close.rolling(params.trend_ma, min_periods=params.trend_ma).mean()
        uptrend = close > trend_ma

        ind = pd.DataFrame(index=data.index)
        ind["queue_fill"] = queue_fill
        ind["capacity"] = capacity
        ind["calm"] = calm.astype(float)
        ind["mom_rank"] = mom_rank
        ind["uptrend"] = uptrend.astype(float)
        return ind

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: CapacityOverflowParams,
    ) -> SignalFrame:
        idx = data.index
        n = len(idx)

        calm = indicators["calm"].fillna(0.0).to_numpy() > 0.5
        uptrend = indicators["uptrend"].fillna(0.0).to_numpy() > 0.5
        mom_rank = indicators["mom_rank"].fillna(0.0).to_numpy()

        # Raw entry condition: calm regime + uptrend + upper-percentile momentum.
        raw = calm & uptrend & (mom_rank >= float(params.entry_rank_thr))

        # Two-bar confirmation: the raw condition must hold this bar and the prior bar.
        confirmed = np.zeros(n, dtype=bool)
        if n > 1:
            confirmed[1:] = raw[1:] & raw[:-1]

        # Fixed-bar exit: hold exactly hold_bars bars, no signal-based exit.
        hold = max(int(params.hold_bars), 1)
        signal = np.zeros(n, dtype=int)
        bars_left = 0
        for i in range(n):
            if bars_left > 0:
                signal[i] = 1
                bars_left -= 1
            elif confirmed[i]:
                signal[i] = 1
                bars_left = hold - 1

        df = pd.DataFrame(index=idx)
        df["signal"] = signal
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = float(max(params.base_size, 0.01))

        return SignalFrame(data=df, signal_column="signal", size_column="size")
