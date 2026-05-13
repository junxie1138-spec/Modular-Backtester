from __future__ import annotations

from enum import Enum, IntEnum


class OrderType(str, Enum):
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"


class OrderSide(str, Enum):
    BUY = "buy"
    SELL = "sell"


class OrderStatus(str, Enum):
    PENDING = "pending"
    FILLED = "filled"
    CANCELLED = "cancelled"
    EXPIRED = "expired"


class SignalDirection(IntEnum):
    FLAT = 0
    LONG = 1
    SHORT = -1
