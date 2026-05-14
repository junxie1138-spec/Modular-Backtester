from __future__ import annotations

from backtester.core.exceptions import (
    BacktesterError,
    ExecutionError,
    ShortNotAllowedError,
)


def test_short_not_allowed_inherits_execution_error():
    assert issubclass(ShortNotAllowedError, ExecutionError)
    assert issubclass(ShortNotAllowedError, BacktesterError)


def test_short_not_allowed_carries_message():
    e = ShortNotAllowedError("shorts disabled at bar 42")
    assert "shorts disabled" in str(e)
    assert "42" in str(e)
