from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
import pytest

from backtester.core.types import SignalFrame
from backtester.strategies.base import BaseStrategy
from backtester.strategies.registry import (
    STRATEGY_REGISTRY,
    get_strategy_class,
    register_strategy,
)


@dataclass(slots=True)
class _FakeParams:
    x: int = 1


class _FakeStrategy(BaseStrategy[_FakeParams]):
    strategy_id = "fake_test_only"

    @classmethod
    def params_type(cls):
        return _FakeParams

    def indicators(self, data, params):
        return pd.DataFrame(index=data.index)

    def generate_signals(self, data, indicators, ctx, params):
        return SignalFrame(data=pd.DataFrame({"signal": [0] * len(data)}, index=data.index))


def test_registry_is_dict():
    assert isinstance(STRATEGY_REGISTRY, dict)


def test_register_and_lookup(monkeypatch):
    monkeypatch.setitem(STRATEGY_REGISTRY, "fake_test_only", _FakeStrategy)
    assert get_strategy_class("fake_test_only") is _FakeStrategy


def test_lookup_unknown_raises():
    with pytest.raises(KeyError, match="unknown_strategy"):
        get_strategy_class("unknown_strategy")


def test_register_strategy_helper(monkeypatch):
    monkeypatch.setattr(
        "backtester.strategies.registry.STRATEGY_REGISTRY", {}, raising=True
    )
    register_strategy(_FakeStrategy)
    from backtester.strategies.registry import STRATEGY_REGISTRY as R
    assert R["fake_test_only"] is _FakeStrategy
