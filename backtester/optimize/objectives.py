from __future__ import annotations

from typing import Callable, Dict


def _sharpe(s: Dict) -> float:
    return float(s.get("sharpe", 0.0))


def _sortino(s: Dict) -> float:
    return float(s.get("sortino", 0.0))


def _total_return(s: Dict) -> float:
    return float(s.get("total_return", 0.0))


def _annualized_return(s: Dict) -> float:
    return float(s.get("annualized_return", 0.0))


def _calmar(s: Dict) -> float:
    dd = abs(float(s.get("max_drawdown", 0.0)))
    if dd == 0:
        return 0.0
    return float(s.get("annualized_return", 0.0)) / dd


OBJECTIVES: Dict[str, Callable[[Dict], float]] = {
    "sharpe": _sharpe,
    "sortino": _sortino,
    "total_return": _total_return,
    "annualized_return": _annualized_return,
    "calmar": _calmar,
}


def resolve_objective(name: str) -> Callable[[Dict], float]:
    if name not in OBJECTIVES:
        raise KeyError(f"unknown objective {name!r}, allowed: {sorted(OBJECTIVES)}")
    return OBJECTIVES[name]
