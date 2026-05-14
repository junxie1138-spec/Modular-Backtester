from __future__ import annotations

from typing import Optional
import pandas as pd

from backtester.config.models import ExecutionConfig
from backtester.engine.fills import Fill, FillEngine
from backtester.engine.orders import Order


class Broker:
    """Thin adapter that owns a FillEngine plus execution policy state."""

    def __init__(self, config: ExecutionConfig):
        self.config = config
        self.fills = FillEngine(
            commission_bps=config.commission_bps,
            slippage_bps=config.slippage_bps,
        )
        self.allow_fractional = config.allow_fractional
        self.allow_short = config.allow_short

    def round_qty(self, qty: float) -> float:
        if self.allow_fractional:
            return qty
        return float(int(qty))

    def submit(self, order: Order, bar: pd.Series) -> Optional[Fill]:
        return self.fills.try_fill(order, bar)
