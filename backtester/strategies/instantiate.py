from __future__ import annotations

from dataclasses import fields
from typing import Any, Dict, Tuple

from backtester.core.exceptions import StrategyError
from backtester.strategies.base import BaseStrategy
from backtester.strategies.registry import get_strategy_class


def build_strategy_and_params(strategy_id: str, params_dict: Dict[str, Any]) -> Tuple[BaseStrategy, Any]:
    cls = get_strategy_class(strategy_id)
    params_type = cls.params_type()
    allowed = {f.name for f in fields(params_type)}
    unknown = set(params_dict) - allowed
    if unknown:
        raise StrategyError(
            f"unknown params for {strategy_id!r}: {sorted(unknown)} (allowed: {sorted(allowed)})"
        )
    params = params_type(**params_dict)
    return cls(), params
