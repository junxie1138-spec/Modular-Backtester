from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

import pandas as pd


class TSPhase(Enum):
    """Tranche-stop phase machine.

    DISARMED -> (reset) -> HARD -> (promote_to_runner) -> RUNNER -> (disarm) -> DISARMED
    """
    HARD = "hard"
    RUNNER = "runner"
    DISARMED = "disarmed"


@dataclass
class TrancheStopState:
    """v0.4.0 close-basis trailing stop with two-phase HARD->RUNNER machine.

    Separate from v0.3.0's TrailingStopState. The two coexist; runs use one
    or the other (config-validation rule 2 enforces the mutex).

    Key semantics:
      - HARD phase: fixed stop at entry_price - hard_stop_atr_mult * atr_at_entry.
        ATR is snapshotted at entry - does NOT trail with current ATR.
      - RUNNER phase: trail at peak_close - runner_atr_mult * atr_now,
        optionally floored at entry_price (breakeven_floor=True by default).
      - Intrabar wicks DO NOT move the runner trail. Only confirmed closes do.
    """

    # configuration (immutable per run)
    hard_stop_atr_mult: float
    runner_atr_mult: float
    breakeven_floor: bool = True
    atr_series: Optional[pd.Series] = None

    # snapshotted at reset() - frozen for the life of the position
    entry_price: float = 0.0
    entry_bar_idx: int = -1
    atr_at_entry: float = float("nan")

    # mutating state - tracks the position
    phase: TSPhase = TSPhase.DISARMED
    peak_close: float = 0.0
    trough_close: float = float("inf")

    # ---- state-machine API ----

    def reset(self, *, entry_price: float, bar_idx: int) -> None:
        """Called on flat -> non-flat transition. Snapshots entry state."""
        self.entry_price = entry_price
        self.entry_bar_idx = bar_idx
        self.atr_at_entry = (
            float(self.atr_series.iloc[bar_idx]) if self.atr_series is not None else float("nan")
        )
        self.peak_close = entry_price
        self.trough_close = entry_price
        self.phase = TSPhase.HARD

    def promote_to_runner(self) -> None:
        """Called by simulator on detected partial close from HARD. Idempotent."""
        if self.phase is TSPhase.HARD:
            self.phase = TSPhase.RUNNER

    def disarm(self) -> None:
        """Called on any transition to flat. Clears mutating state."""
        self.phase = TSPhase.DISARMED
        self.peak_close = 0.0
        self.trough_close = float("inf")

    def update(self, bar: pd.Series) -> None:
        """Per-bar ratchet on CLOSE only. Intrabar wicks ignored."""
        if self.phase is TSPhase.DISARMED:
            return
        c = float(bar["close"])
        if c > self.peak_close:
            self.peak_close = c
        if c < self.trough_close:
            self.trough_close = c

    def stop_price(self, *, sign: int, bar_idx: int) -> Optional[float]:
        """Stop level for the next bar's STOP order. Implemented in Task 19."""
        raise NotImplementedError("Implemented in Task 19")
