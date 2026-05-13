from __future__ import annotations

import pytest

from backtester.optimize.objectives import resolve_objective, OBJECTIVES


def test_known_objectives():
    for name in ["sharpe", "sortino", "total_return", "calmar", "annualized_return"]:
        assert name in OBJECTIVES


def test_resolve_returns_callable_score():
    fn = resolve_objective("sharpe")
    summary = {"sharpe": 1.5, "max_drawdown": -0.1, "annualized_return": 0.1}
    assert fn(summary) == 1.5


def test_calmar_uses_abs_drawdown():
    fn = resolve_objective("calmar")
    s = {"annualized_return": 0.2, "max_drawdown": -0.1, "sharpe": 0, "sortino": 0}
    assert fn(s) == pytest.approx(2.0)


def test_unknown_objective_raises():
    with pytest.raises(KeyError, match="unknown"):
        resolve_objective("nope")
