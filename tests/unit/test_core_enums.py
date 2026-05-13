from backtester.core.enums import OrderType, OrderSide, SignalDirection
from backtester.core.exceptions import (
    BacktesterError, ConfigError, DataError, StrategyError,
)


def test_order_types_have_expected_values():
    assert OrderType.MARKET.value == "market"
    assert OrderType.LIMIT.value == "limit"
    assert OrderType.STOP.value == "stop"


def test_order_sides():
    assert OrderSide.BUY.value == "buy"
    assert OrderSide.SELL.value == "sell"


def test_signal_directions():
    assert SignalDirection.FLAT.value == 0
    assert SignalDirection.LONG.value == 1


def test_exceptions_inherit_from_base():
    for exc in (ConfigError, DataError, StrategyError):
        assert issubclass(exc, BacktesterError)


def test_exceptions_carry_message():
    e = ConfigError("bad key")
    assert "bad key" in str(e)
