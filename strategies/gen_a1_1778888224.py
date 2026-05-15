from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class MomentumPercentileParams:
    mom_lookback: int = 21
    rank_window: int = 252
    entry_pct: float = 0.80
    spike_lookback: int = 252
    spike_pct: float = 0.95
    refractory_bars: int = 5
    atr_window: int = 14
    trail_k: float = 3.0


class GeneratedStrategy(BaseStrategy[MomentumPercentileParams]):
    strategy_id = "gen_a1_1778888224"

    @classmethod
    def params_type(cls) -> type[MomentumPercentileParams]:
        return MomentumPercentileParams

    @staticmethod
    def warmup_bars(params: MomentumPercentileParams) -> int:
        return int(max(
            params.mom_lookback + params.rank_window + 1,
            params.spike_lookback + 1,
            params.atr_window + 1,
        ))

    @staticmethod
    def indicators(data: pd.DataFrame, params: MomentumPercentileParams) -> pd.DataFrame:
        close = data["close"]
        high = data["high"]
        low = data["low"]

        # Momentum measured as a self-normalizing rolling percentile rank.
        mom = close.pct_change(params.mom_lookback)
        mom_rank = mom.rolling(params.rank_window).rank(pct=True)

        # Single-bar return spike intensity, also as a rolling percentile rank.
        ret1 = close.pct_change()
        spike_rank = ret1.abs().rolling(params.spike_lookback).rank(pct=True)

        # Average true range for the ratcheting trailing stop.
        prev_close = close.shift(1)
        tr = pd.concat([
            (high - low),
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ], axis=1).max(axis=1)
        atr = tr.rolling(params.atr_window).mean()

        out = pd.DataFrame(index=data.index)
        out["mom_rank"] = mom_rank
        out["spike_rank"] = spike_rank
        out["atr"] = atr
        return out

    @staticmethod
    def generate_signals(
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: MomentumPercentileParams,
    ) -> SignalFrame:
        n = len(data)
        close = data["close"].to_numpy(dtype=float)
        mom_rank = indicators["mom_rank"].to_numpy(dtype=float)
        spike_rank = indicators["spike_rank"].to_numpy(dtype=float)
        atr = indicators["atr"].to_numpy(dtype=float)

        signal = np.zeros(n, dtype=int)
        position = 0
        high_water = np.nan
        # Start unlocked: no spike has occurred yet.
        bars_since_spike = int(params.refractory_bars)

        for i in range(n):
            # Update the post-spike refractory counter from this bar's reading.
            sr = spike_rank[i]
            if np.isfinite(sr) and sr >= params.spike_pct:
                bars_since_spike = 0
            else:
                bars_since_spike += 1
            locked = bars_since_spike < params.refractory_bars

            if position == 0:
                cur = mom_rank[i]
                prev = mom_rank[i - 1] if i > 0 else np.nan
                # Fresh upward crossing of the momentum percentile threshold.
                cross = (
                    np.isfinite(cur) and np.isfinite(prev)
                    and cur >= params.entry_pct and prev < params.entry_pct
                )
                tradable = np.isfinite(atr[i]) and atr[i] > 0.0
                if cross and tradable and not locked:
                    position = 1
                    high_water = close[i]
                    signal[i] = 1
                else:
                    signal[i] = 0
            else:
                # Ratchet the in-trade high-water mark up only.
                if close[i] > high_water:
                    high_water = close[i]
                a = atr[i]
                stop_hit = (
                    np.isfinite(a)
                    and close[i] <= high_water - params.trail_k * a
                )
                if stop_hit:
                    position = 0
                    high_water = np.nan
                    signal[i] = 0
                else:
                    signal[i] = 1

        df = pd.DataFrame(index=data.index)
        df["signal"] = signal
        # Decide on bar N's close, fill on bar N+1.
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = 1.0
        return SignalFrame(data=df, signal_column="signal", size_column="size")
