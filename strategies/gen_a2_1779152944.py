from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class PredatorPreyGapParams:
    atr_period: int = 14
    prey_span: int = 8
    predator_span: int = 8
    predator_slow_window: int = 30
    predator_decline_bars: int = 3
    gap_atr_frac: float = 0.15
    trail_k: float = 2.5
    max_hold: int = 10


class GeneratedStrategy(BaseStrategy[PredatorPreyGapParams]):
    """Regime-switching long-only gap strategy.

    Models overnight gaps as a Lotka-Volterra predator-prey system:
    down-gap pressure is the predator population, up-gap pressure is the prey.
    Enters only when BOTH primitives agree on the same bar:
      (1) gap primitive  - a fresh up-gap large relative to ATR;
      (2) regime primitive - the predator population has crashed (below its
          slow average AND declining) while the prey population is rising.
    Exits via an ATR rolling-high trailing stop that only ratchets up.
    """

    strategy_id = "gen_a2_1779152944"

    @classmethod
    def params_type(cls):
        return PredatorPreyGapParams

    @staticmethod
    def warmup_bars(params: PredatorPreyGapParams) -> int:
        base = max(
            params.atr_period,
            params.predator_slow_window,
            params.prey_span,
            params.predator_span,
        )
        return int(base + params.predator_decline_bars + 5)

    @staticmethod
    def indicators(data: pd.DataFrame, params: PredatorPreyGapParams) -> pd.DataFrame:
        close = data["close"]
        high = data["high"]
        low = data["low"]
        open_ = data["open"]
        prior_close = close.shift(1)

        # True range / ATR (NaN during warmup, handled downstream).
        tr = pd.concat(
            [
                high - low,
                (high - prior_close).abs(),
                (low - prior_close).abs(),
            ],
            axis=1,
        ).max(axis=1)
        atr = tr.rolling(params.atr_period).mean()

        # Overnight gap as a fraction of prior close.
        gap_pct = ((open_ - prior_close) / prior_close).fillna(0.0)

        # Predator-prey populations derived from gap polarity.
        prey_food = gap_pct.clip(lower=0.0)         # up-gap magnitude (prey)
        predator_food = (-gap_pct).clip(lower=0.0)  # down-gap magnitude (predator)
        prey = prey_food.ewm(span=params.prey_span, adjust=False).mean()
        predator = predator_food.ewm(span=params.predator_span, adjust=False).mean()
        predator_slow = predator.rolling(params.predator_slow_window).mean()

        # Predator population has crashed: below its slow average and falling.
        pred_falling = (predator.diff() < 0).rolling(
            params.predator_decline_bars
        ).sum()
        predator_crashed = (predator < predator_slow) & (
            pred_falling >= params.predator_decline_bars
        )
        prey_rising = prey > prey.shift(1)
        regime = predator_crashed & prey_rising

        # Gap primitive: a fresh up-gap large relative to ATR.
        atr_frac = (atr / prior_close).fillna(0.0)
        up_gap = (gap_pct > 0.0) & (gap_pct >= (params.gap_atr_frac * atr_frac))

        # Two-primitive AND: both must agree on the same bar.
        entry_ok = (regime & up_gap).fillna(False)

        out = pd.DataFrame(index=data.index)
        out["atr"] = atr
        out["entry_ok"] = entry_ok.astype(float)
        return out

    @staticmethod
    def generate_signals(
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: PredatorPreyGapParams,
    ) -> SignalFrame:
        close = data["close"].to_numpy(dtype=float)
        atr = indicators["atr"].to_numpy(dtype=float)
        entry_ok = indicators["entry_ok"].to_numpy(dtype=float)
        n = len(close)
        signal = np.zeros(n, dtype=int)

        position = 0
        high_water = 0.0
        bars_held = 0
        for i in range(n):
            if position == 0:
                if (
                    entry_ok[i] >= 1.0
                    and not np.isnan(atr[i])
                    and atr[i] > 0.0
                ):
                    position = 1
                    high_water = close[i]
                    bars_held = 0
                    signal[i] = 1
            else:
                bars_held += 1
                # Rolling-high water mark only ratchets up.
                if close[i] > high_water:
                    high_water = close[i]
                stop_level = high_water - params.trail_k * atr[i]
                exit_now = (
                    np.isnan(atr[i])
                    or close[i] <= stop_level
                    or bars_held >= params.max_hold
                )
                if exit_now:
                    position = 0
                    signal[i] = 0
                else:
                    signal[i] = 1

        df = pd.DataFrame(index=data.index)
        df["signal"] = signal
        df["size"] = 1.0
        # MANDATORY one-bar shift: decide on bar N close, fill on bar N+1.
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")
