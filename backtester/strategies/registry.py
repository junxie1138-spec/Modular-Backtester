from __future__ import annotations

from typing import Dict, Type

from backtester.strategies.base import BaseStrategy

STRATEGY_REGISTRY: Dict[str, Type[BaseStrategy]] = {}


def register_strategy(cls: Type[BaseStrategy]) -> Type[BaseStrategy]:
    """Register a strategy class by its `strategy_id`."""
    if not getattr(cls, "strategy_id", None):
        raise ValueError(f"{cls.__name__} is missing a non-empty `strategy_id`")
    STRATEGY_REGISTRY[cls.strategy_id] = cls
    return cls


def get_strategy_class(strategy_id: str) -> Type[BaseStrategy]:
    if strategy_id not in STRATEGY_REGISTRY:
        raise KeyError(
            f"Strategy {strategy_id!r} is not registered. "
            f"Known: {sorted(STRATEGY_REGISTRY)}"
        )
    return STRATEGY_REGISTRY[strategy_id]


# --- Default strategy registrations (explicit, predictable order) ---
from strategies.sma_cross import SMACrossStrategy  # noqa: E402
from strategies.rsi_mean_reversion import RSIMeanReversionStrategy  # noqa: E402
from strategies.breakout_20d import Breakout20DStrategy  # noqa: E402
from strategies.rsi_long_short import RSILongShortStrategy  # noqa: E402

register_strategy(SMACrossStrategy)
register_strategy(RSIMeanReversionStrategy)
register_strategy(Breakout20DStrategy)
register_strategy(RSILongShortStrategy)
