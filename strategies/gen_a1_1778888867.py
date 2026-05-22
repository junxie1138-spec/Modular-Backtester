from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


# Fixed structural window (NOT tunable - the twist caps tunable params at 2).
# One trading week is treated as the natural "tidal period" of the overnight
# gap flow used for both the velocity and the acceleration measurements.
_ROC_WINDOW = 5


@dataclass(slots=True)
class TideTurnParams:
    # Exit at +profit_target gain.
    profit_target: float = 0.04
    # ...or after time_stop bars held, whichever comes first.
    time_stop: int = 10


class GeneratedStrategy(BaseStrategy[TideTurnParams]):
    strategy_id = "gen_a1_1778888867"

    @classmethod
    def params_type(cls):
        return TideTurnParams

    @staticmethod
    def warmup_bars(params: TideTurnParams) -> int:
        # gap needs 1 prior bar; velocity = pct_change(w); acceleration =
        # velocity.diff(w); plus a 1-bar shift for the cross detection.
        # 2*_ROC_WINDOW + 2 = 12; round up to 20 for headroom.
        return 20

    @staticmethod
    def indicators(data: pd.DataFrame, params: TideTurnParams) -> pd.DataFrame:
        w = _ROC_WINDOW
        close = data["close"]
        open_ = data["open"]

        # Overnight gap = the tidal pull between sessions. First bar has no
        # prior close -> NaN -> treat as zero gap.
        gap = (open_ / close.shift(1) - 1.0).fillna(0.0)

        # Cumulative overnight-gap index G = the running "tide level": an
        # equity curve built only from overnight moves. cumprod of (1+0)=1
        # at the warmup edge, so G is NaN-free.
        tide_index = (1.0 + gap).cumprod()

        # Velocity = rate of change of the tide level over one tidal period.
        velocity = tide_index.pct_change(w)

        # Acceleration = rate-of-change acceleration (2nd order) of the tide.
        acceleration = velocity.diff(w)

        velocity = velocity.fillna(0.0)
        acceleration = acceleration.fillna(0.0)

        # Turn of the tide: acceleration freshly crosses UP through zero while
        # the tide is still receding (velocity < 0) -> slack water before the
        # flood. This is the long-only entry trigger.
        accel_prev = acceleration.shift(1).fillna(0.0)
        entry = (
            (acceleration > 0.0)
            & (accel_prev <= 0.0)
            & (velocity < 0.0)
        )

        out = pd.DataFrame(index=data.index)
        out["gap"] = gap
        out["tide_index"] = tide_index
        out["velocity"] = velocity
        out["acceleration"] = acceleration
        out["entry"] = entry.astype(float)
        return out

    @staticmethod
    def generate_signals(
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: TideTurnParams,
    ) -> SignalFrame:
        n = len(data)
        close = data["close"].to_numpy(dtype=float)
        entry = indicators["entry"].fillna(0.0).to_numpy() > 0.5

        signal = np.zeros(n, dtype=int)
        pt = float(params.profit_target)
        ts = int(params.time_stop)
        if ts < 1:
            ts = 1

        in_pos = False
        entry_idx = -1
        entry_price = 0.0

        for i in range(n):
            if in_pos:
                bars_held = i - entry_idx
                gain = (close[i] / entry_price - 1.0) if entry_price > 0.0 else 0.0
                # profit-target OR time-stop, whichever fires first.
                if gain >= pt or bars_held >= ts:
                    in_pos = False
                    entry_idx = -1
                    entry_price = 0.0
                    signal[i] = 0
                else:
                    signal[i] = 1
            else:
                if entry[i]:
                    in_pos = True
                    entry_idx = i
                    entry_price = close[i]
                    signal[i] = 1
                else:
                    signal[i] = 0

        df = pd.DataFrame(index=data.index)
        df["signal"] = signal
        # MANDATORY one-bar shift: decide on bar N's close, fill on N+1.
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = 1.0
        return SignalFrame(data=df, signal_column="signal", size_column="size")
