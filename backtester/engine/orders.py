from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional
import pandas as pd

from backtester.core.enums import OrderSide, OrderStatus, OrderType


@dataclass
class Order:
    timestamp: pd.Timestamp
    symbol: str
    side: OrderSide
    qty: float
    order_type: OrderType
    limit_price: Optional[float] = None
    stop_price: Optional[float] = None
    status: OrderStatus = OrderStatus.PENDING
    tag: str = ""

    def __post_init__(self) -> None:
        if self.qty <= 0:
            raise ValueError("Order qty must be > 0")
        if self.order_type == OrderType.LIMIT and self.limit_price is None:
            raise ValueError("LIMIT order requires limit_price")
        if self.order_type == OrderType.STOP and self.stop_price is None:
            raise ValueError("STOP order requires stop_price")
