from __future__ import annotations

import logging
from dataclasses import asdict, is_dataclass
from typing import Any, Dict, List, Tuple, Type

import pandas as pd

from backtester.core.types import BacktestResult
from backtester.engine.backtest_engine import BacktestEngine
from backtester.optimize.objectives import resolve_objective
from backtester.optimize.parameter_space import expand_grid
from backtester.strategies.base import BaseStrategy

log = logging.getLogger("backtester.optimize")


class GridSearchOptimizer:
    def __init__(self, engine: BacktestEngine, objective: str = "sharpe"):
        self.engine = engine
        self.objective_name = objective
        self.score_fn = resolve_objective(objective)

    def find_best(
        self,
        strategy_cls: Type[BaseStrategy],
        data: pd.DataFrame,
        param_space: Dict[str, List[Any]],
        symbol: str,
        timeframe: str,
    ) -> Tuple[Any, BacktestResult, List[Dict]]:
        params_type = strategy_cls.params_type()
        strategy = strategy_cls()

        results: List[Dict] = []
        best: Tuple[float, Any, BacktestResult] | None = None

        for combo in expand_grid(param_space):
            try:
                params = params_type(**combo)
                result = self.engine.run(strategy, data, params, symbol=symbol, timeframe=timeframe)
                score = self.score_fn(result.summary)
            except Exception as exc:
                log.warning("grid combo %s failed: %s", combo, exc)
                results.append({"params": combo, "score": float("-inf"), "summary": {"error": str(exc)}})
                continue

            results.append({
                "params": asdict(params) if is_dataclass(params) else combo,
                "score": float(score),
                "summary": result.summary,
            })
            if best is None or score > best[0]:
                best = (score, params, result)

        if best is None:
            raise RuntimeError("grid search produced no successful runs")
        return best[1], best[2], results
