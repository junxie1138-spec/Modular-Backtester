from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import pandas as pd


@dataclass
class TrailingStopState:
    """Singleton per-simulation trailing-stop state.

    Owned by `PortfolioSimulator`. Reset on entry / flip; updated each
    bar with the bar's high/low; queried for the next bar's stop price.
    """

    pct: Optional[float] = None
    atr_mult: Optional[float] = None
    atr_series: Optional[pd.Series] = None  # aligned to data.index; required iff atr_mult
    peak_high: float = 0.0
    trough_low: float = float("inf")
    armed: bool = False

    @property
    def enabled(self) -> bool:
        return self.pct is not None or self.atr_mult is not None

    def reset(self, entry_price: float) -> None:
        self.peak_high = entry_price
        self.trough_low = entry_price
        self.armed = True

    def disarm(self) -> None:
        self.armed = False
        self.peak_high = 0.0
        self.trough_low = float("inf")

    def update(self, high: float, low: float) -> None:
        if not self.armed:
            return
        if high > self.peak_high:
            self.peak_high = high
        if low < self.trough_low:
            self.trough_low = low

    def stop_price(self, sign: int, bar_idx: int) -> Optional[float]:
        if not self.armed or sign == 0:
            return None
        if self.pct is not None:
            if sign > 0:
                return self.peak_high * (1.0 - self.pct)
            return self.trough_low * (1.0 + self.pct)
        # ATR mode handled in Task 6
        return None
