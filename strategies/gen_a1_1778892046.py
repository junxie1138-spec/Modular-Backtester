from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class YieldStreakParams:
    yield_streak: int = 3
    sma_period: int = 100
    atr_period: int = 14
    breakeven_pct: float = 0.02
    trail_atr_mult: float = 2.5
    init_stop_atr_mult: float = 2.0
    time_stop: int = 5
    strain_norm: float = 3.0
    base_size: float = 0.4
    max_size: float = 1.0


class GeneratedStrategy(BaseStrategy[YieldStreakParams]):
    """Trend-strength strategy: a consecutive up-close streak is treated as a
    material under stress. Below the yield-point count the deformation is
    elastic (no trade); when the streak reaches the yield count the trend has
    plastically deformed and a long is opened. Position size scales with the
    'plastic strain' - the ATR-normalised price gain accumulated over the
    streak window. Exit is breakeven-then-trail.
    """

    strategy_id = "gen_a1_1778892046"

    @classmethod
    def params_type(cls):
        return YieldStreakParams

    @staticmethod
    def warmup_bars(params: YieldStreakParams) -> int:
        return int(params.sma_period + params.yield_streak + 1)

    @staticmethod
    def indicators(data: pd.DataFrame, params: YieldStreakParams) -> pd.DataFrame:
        close = data["close"]
        high = data["high"]
        low = data["low"]

        # Consecutive up-close streak count (vectorised; resets to 0 on any
        # non-up bar). Each group begins with the resetting non-up bar, so
        # cumcount() yields 0 for that bar and 1,2,3,... for the up-run.
        up = close > close.shift(1)
        grp = (~up).cumsum()
        streak = up.groupby(grp).cumcount()
        streak = streak.where(up, 0).astype(int)

        # Average true range (NaN during the first atr_period-1 bars).
        prev_close = close.shift(1)
        tr = pd.concat(
            [
                high - low,
                (high - prev_close).abs(),
                (low - prev_close).abs(),
            ],
            axis=1,
        ).max(axis=1)
        atr = tr.rolling(params.atr_period, min_periods=params.atr_period).mean()

        # Long regime filter.
        sma = close.rolling(params.sma_period, min_periods=params.sma_period).mean()

        out = pd.DataFrame(index=data.index)
        out["streak"] = streak
        out["atr"] = atr
        out["sma"] = sma
        return out

    @staticmethod
    def generate_signals(
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: YieldStreakParams,
    ) -> SignalFrame:
        close = data["close"].to_numpy(dtype=float)
        high = data["high"].to_numpy(dtype=float)
        low = data["low"].to_numpy(dtype=float)
        streak = indicators["streak"].to_numpy()
        atr = indicators["atr"].to_numpy(dtype=float)
        sma = indicators["sma"].to_numpy(dtype=float)
        n = len(close)

        warmup = int(params.sma_period + params.yield_streak + 1)
        size_span = max(params.max_size - params.base_size, 0.0)
        strain_norm = params.strain_norm if params.strain_norm > 0.0 else 1.0

        sig = np.zeros(n, dtype=int)
        size = np.ones(n, dtype=float)

        in_pos = False
        entry_price = 0.0
        stop = 0.0
        breakeven_armed = False
        bars_held = 0
        entry_size = params.base_size

        for i in range(n):
            if i < warmup or np.isnan(atr[i]) or np.isnan(sma[i]):
                continue

            if not in_pos:
                # Plastic-yield entry: the up-close streak first reaches the
                # yield-point count, in a long regime. Entering strictly at
                # streak == yield_streak gives a single deterministic entry
                # per run (no churn) and a fixed-length strain window.
                if (
                    streak[i] == params.yield_streak
                    and close[i] > sma[i]
                    and atr[i] > 0.0
                ):
                    streak_gain = close[i] - close[i - params.yield_streak]
                    strain = max(streak_gain, 0.0) / atr[i]
                    frac = strain / strain_norm
                    if frac < 0.0:
                        frac = 0.0
                    elif frac > 1.0:
                        frac = 1.0
                    entry_size = params.base_size + frac * size_span
                    entry_price = close[i]
                    stop = close[i] - params.init_stop_atr_mult * atr[i]
                    breakeven_armed = False
                    bars_held = 0
                    in_pos = True
                    sig[i] = 1
                    size[i] = entry_size
                continue

            # --- position management (path-dependent exit) ---
            bars_held += 1

            # Breakeven: once price has run +breakeven_pct, lift stop to entry.
            if (
                not breakeven_armed
                and high[i] >= entry_price * (1.0 + params.breakeven_pct)
            ):
                if entry_price > stop:
                    stop = entry_price
                breakeven_armed = True

            # Trail only after breakeven is armed; stop only ever moves up.
            if breakeven_armed:
                trail = close[i] - params.trail_atr_mult * atr[i]
                if trail > stop:
                    stop = trail

            exit_now = low[i] <= stop or bars_held >= params.time_stop
            if exit_now:
                in_pos = False
                sig[i] = 0
            else:
                sig[i] = 1
                size[i] = entry_size

        df = pd.DataFrame(index=data.index)
        df["signal"] = sig
        df["size"] = size
        # Mandatory one-bar shift: decide on bar N close, fill on N+1.
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = df["size"].shift(1).fillna(1.0)
        return SignalFrame(data=df, signal_column="signal", size_column="size")
