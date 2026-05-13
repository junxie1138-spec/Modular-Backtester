from __future__ import annotations


class BacktesterError(Exception):
    """Base exception for the backtester framework."""


class ConfigError(BacktesterError):
    """Raised when a config is malformed or invalid."""


class DataError(BacktesterError):
    """Raised when input data is missing or invalid."""


class StrategyError(BacktesterError):
    """Raised when a strategy violates its contract."""


class ExecutionError(BacktesterError):
    """Raised when the broker / portfolio simulator cannot proceed."""
