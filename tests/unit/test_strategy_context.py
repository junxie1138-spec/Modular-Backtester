from __future__ import annotations

import dataclasses
import inspect

import pytest

from backtester.core.types import StrategyContext


def test_strategy_context_has_v04_fields():
    """v0.4.0 fields exist on StrategyContext with safe defaults."""
    field_names = {f.name for f in dataclasses.fields(StrategyContext)}
    assert "position_phase" in field_names
    assert "bars_in_phase" in field_names
    assert "recent_pnl" in field_names
    assert "regime" in field_names


def test_strategy_context_defaults_safe_for_v03_callers():
    """v0.3.0 call-sites that don't pass v0.4.0 fields still work."""
    sig = inspect.signature(StrategyContext)
    # Build kwargs for required fields only (those without defaults).
    required_kwargs = {}
    for name, param in sig.parameters.items():
        if param.default is inspect.Parameter.empty:
            ann = param.annotation
            if ann is str:
                required_kwargs[name] = ""
            elif ann is int:
                required_kwargs[name] = 0
            else:
                required_kwargs[name] = None
    ctx = StrategyContext(**required_kwargs)
    # The v0.4.0 fields exist with safe defaults (empty dict, None).
    assert ctx.position_phase == {} or ctx.position_phase is None
    assert ctx.bars_in_phase == {} or ctx.bars_in_phase is None
