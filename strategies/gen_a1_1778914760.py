from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class GapStreakParams:
    gap_eps_bps: float = 5.0
    entry_streak: int = 3
    hold_bars: int = 8
    sma_period: int = 200
    max_scale: float = 2.0
    use_regime: bool = True


class GeneratedStrategy(BaseStrategy[GapStreakParams]):
    strategy_id = "gen_a1_1778914760"

    @classmethod
    def params_type(cls) -> type[GapStreakParams]:
        return GapStreakParams

    @staticmethod
    def warmup_bars(params: GapStreakParams) -> int:
        # rolling SMA window + one bar for the prior-close gap reference
        return int(params.sma_period) + 2

    @staticmethod
    def indicators(data: pd.DataFrame, params: GapStreakParams) -> pd.DataFrame:
        out = pd.DataFrame(index=data.index)

        prev_close = data["close"].shift(1)
        gap_pct = data["open"] / prev_close - 1.0
        out["gap_pct"] = gap_pct

        # Symmetric deadband (hysteresis): tiny gaps inside +/- eps are
        # neutral - they neither extend nor reset the streak.
        eps = max(float(params.gap_eps_bps), 0.0) / 10000.0
        up_gap = (gap_pct > eps).fillna(False)
        down_gap = (gap_pct < -eps).fillna(False)

        # Consecutive up-gap streak: increments on an up-gap, carries through
        # neutral gaps, resets to 0 on a down-gap. cumsum-within-group trick
        # keeps this fully vectorised and NaN-safe.
        increment = up_gap.astype(int)
        group_id = down_gap.astype(int).cumsum()
        streak = increment.groupby(group_id).cumsum()
        out["streak"] = streak.astype(float)

        sma_period = max(int(params.sma_period), 1)
        sma = data["close"].rolling(sma_period, min_periods=sma_period).mean()
        out["sma"] = sma
        if params.use_regime:
            out["regime"] = (data["close"] > sma).fillna(False)
        else:
            out["regime"] = True

        return out

    @staticmethod
    def generate_signals(
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: GapStreakParams,
    ) -> SignalFrame:
        n = len(data)

        streak = np.nan_to_num(
            indicators["streak"].to_numpy(dtype=float), nan=0.0
        )
        regime = indicators["regime"].to_numpy(dtype=bool)

        entry_streak = max(int(params.entry_streak), 1)
        hold_bars = max(int(params.hold_bars), 1)
        max_scale = max(float(params.max_scale), 1.0)

        entry_ok = (streak >= entry_streak) & regime

        # Signal-scaled sizing: deeper streaks express more conviction.
        scale = np.clip(streak / float(entry_streak), 1.0, max_scale)

        signal = np.zeros(n, dtype=int)
        size = np.ones(n, dtype=float)

        # Fixed-bar exit: hold exactly hold_bars bars from entry, no
        # re-entry while in the position, then release.
        i = 0
        while i < n:
            if entry_ok[i]:
                end = min(i + hold_bars, n)
                s = float(scale[i])
                signal[i:end] = 1
                size[i:end] = s
                i = end
            else:
                i += 1

        df = pd.DataFrame(index=data.index)
        df["signal"] = signal
        df["size"] = size

        # Mandatory one-bar shift: decide on bar N close, fill on bar N+1.
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = df["size"].shift(1).fillna(1.0)
        df["size"] = df["size"].clip(lower=1e-6)

        return SignalFrame(data=df, signal_column="signal", size_column="size")
