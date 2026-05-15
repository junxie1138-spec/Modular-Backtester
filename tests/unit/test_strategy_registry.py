from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from unittest import mock

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


def test_curated_strategies_registered_on_import() -> None:
    """Importing the registry registers the curated hand-written strategies."""
    import backtester.strategies.registry as reg
    assert reg.get_strategy_class("sma_cross") is not None
    assert reg.get_strategy_class("mean_reversion_atr") is not None


def test_discover_only_imports_gen_modules_in_sorted_order() -> None:
    """discover_generated_strategies imports only strategies.gen_* modules,
    in sorted (deterministic) order."""
    from backtester.strategies.registry import discover_generated_strategies
    imported = discover_generated_strategies()
    assert all(name.startswith("strategies.gen_") for name in imported)
    assert imported == sorted(imported)


def test_discover_invokes_invalidate_caches() -> None:
    """importlib.invalidate_caches() is called before globbing so a
    just-written generated file is visible."""
    import backtester.strategies.registry as reg
    with mock.patch.object(reg.importlib, "invalidate_caches") as inv:
        reg.discover_generated_strategies()
    inv.assert_called_once()


def test_discover_skips_broken_generated_module(caplog) -> None:
    """A gen_*.py that fails to import is skipped (logged with its filename
    and exception), never fatal; valid modules still register."""
    import strategies as strategies_pkg
    from backtester.strategies.registry import discover_generated_strategies
    pkg_dir = Path(strategies_pkg.__file__).resolve().parent
    broken = pkg_dir / "gen_zzz_brokenfixture.py"
    broken.write_text("this is not valid python !!!\n", encoding="utf-8")
    try:
        with caplog.at_level("WARNING"):
            imported = discover_generated_strategies()   # must not raise
    finally:
        broken.unlink(missing_ok=True)
    assert "strategies.gen_zzz_brokenfixture" not in imported
    assert "gen_zzz_brokenfixture.py" in caplog.text
    assert len(imported) > 0   # valid gen_*.py modules still registered
