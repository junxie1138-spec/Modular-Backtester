from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy

# Fixed (non-tunable) structural constants -- kept out of the params class so
# the strategy exposes exactly two tunable knobs.
PEAK_LOOKBACK = 60   # bars used to define the spring's anchor (rolling peak)
HOLD_BARS = 4        # fixed-bar exit: flatten exactly 4 bars after entry


@dataclass(slots=True)
class GapSpringParams:
    gap_thr: float = 0.004   # minimum |overnight gap| to register an impulse
    dd_thr: float = 0.05     # drawdown depth above which the spring is 'taut'


class GeneratedStrategy(BaseStrategy[GapSpringParams]):
    strategy_id = "gen_a1_1778910148"

    @classmethod
    def params_type(cls):
        return GapSpringParams

    def warmup_bars(self, params: GapSpringParams) -> int:
        # rolling peak needs PEAK_LOOKBACK bars; gap uses a 1-bar shift.
        return PEAK_LOOKBACK + 1

    def indicators(self, data: pd.DataFrame, params: GapSpringParams) -> pd.DataFrame:
        close = data["close"]
        open_ = data["open"]

        prev_close = close.shift(1)
        # overnight gap as a return; NaN on the first bar.
        gap = (open_ - prev_close) / prev_close

        # rolling peak = spring anchor; depth = how far price is stretched below it.
        peak = close.rolling(PEAK_LOOKBACK, min_periods=PEAK_LOOKBACK).max()
        depth = (peak - close) / peak  # >= 0 when underwater, ~0 at the peak

        out = pd.DataFrame(index=data.index)
        out["gap"] = gap
        out["depth"] = depth
        return out

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: GapSpringParams,
    ) -> SignalFrame:
        # NaN-safe extraction: warmup NaNs become neutral zeros.
        gap = indicators["gap"].fillna(0.0).to_numpy()
        depth = indicators["depth"].fillna(0.0).to_numpy()
        n = len(data)

        # An entry only exists on a bar with a real overnight impulse.
        trigger = np.abs(gap) >= params.gap_thr
        # Taut spring (deep drawdown) -> restoring force dominates -> fade the gap.
        # Slack spring (near peak)    -> no restoring tension    -> follow the gap.
        taut = depth >= params.dd_thr
        gap_sign = np.sign(gap).astype(int)
        raw_dir = np.where(taut, -gap_sign, gap_sign).astype(int)
        raw_dir = np.where(trigger, raw_dir, 0).astype(int)

        # Fixed-bar exit: hold exactly HOLD_BARS bars, ignore triggers while in a
        # position, then flatten. Path-dependent, so a bar-indexed loop is used.
        final = np.zeros(n, dtype=int)
        position = 0
        bars_held = 0
        for i in range(n):
            if position != 0:
                bars_held += 1
                if bars_held >= HOLD_BARS:
                    position = 0
                    bars_held = 0
                    final[i] = 0
                else:
                    final[i] = position
            else:
                if raw_dir[i] != 0:
                    position = raw_dir[i]
                    bars_held = 0
                    final[i] = position
                else:
                    final[i] = 0

        df = pd.DataFrame(index=data.index)
        df["signal"] = final
        # Mandatory one-bar shift: decide on bar N's close, fill on bar N+1.
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = 1.0
        return SignalFrame(data=df, signal_column="signal", size_column="size")
