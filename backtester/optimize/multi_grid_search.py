from __future__ import annotations

import logging
from dataclasses import asdict, is_dataclass
from typing import Any, Optional

import pandas as pd

from backtester.engine.multi_backtest_engine import MultiSymbolBacktestEngine
from backtester.engine.multi_portfolio import MultiSymbolResult
from backtester.optimize.lhs_sampler import sample_param_space
from backtester.optimize.objectives import resolve_objective
from backtester.optimize.parameter_space import expand_grid
from backtester.strategies.base import BaseStrategy

log = logging.getLogger("backtester.optimize")


class MultiSymbolGridSearchOptimizer:
    """Multi-symbol parameter sweep using MultiSymbolBacktestEngine.

    Supports both 'grid' (full Cartesian) and 'lhs' (discrete Latin-hypercube
    over index positions) sampling modes.
    """

    def __init__(self, engine: MultiSymbolBacktestEngine, objective: str = "sharpe"):
        self.engine = engine
        self.objective_name = objective
        self.score_fn = resolve_objective(objective)

    def find_best(
        self,
        *,
        strategy_cls: type[BaseStrategy],
        symbols: list[str],
        data: dict[str, pd.DataFrame],
        sectors: dict[str, str],
        aux_data: dict[str, pd.DataFrame],
        param_space: dict[str, list[Any]],
        regime_config: Optional[Any] = None,
        sampling: str = "grid",
        random_n: int = 100,
        random_seed: int = 0,
    ) -> tuple[Any, MultiSymbolResult, list[dict]]:
        params_type = strategy_cls.params_type()
        strategy = strategy_cls()

        if sampling == "lhs":
            combos = sample_param_space(space=param_space, random_n=random_n, seed=random_seed)
        else:
            combos = list(expand_grid(param_space))

        results: list[dict] = []
        best: tuple[float, Any, MultiSymbolResult] | None = None

        for i, combo in enumerate(combos):
            try:
                params = params_type(**combo)
                result = self.engine.run(
                    strategy=strategy, symbols=symbols, data=data,
                    sectors=sectors, aux_data=aux_data, params=params,
                    regime_config=regime_config,
                )
                # Score against a summary-style dict so existing objective fns work.
                summary = {
                    "total_return": result.portfolio_total_return,
                    "annualized_return": result.portfolio_total_return,  # crude; OK for ranking
                    "sharpe": result.portfolio_sharpe,
                    "max_drawdown": result.portfolio_max_drawdown,
                    "calmar": (
                        result.portfolio_total_return / abs(result.portfolio_max_drawdown)
                        if result.portfolio_max_drawdown < 0 else 0.0
                    ),
                }
                score = self.score_fn(summary)
            except Exception as exc:
                log.warning("multi-symbol combo %d %s failed: %s", i, combo, exc)
                results.append({"params": combo, "score": float("-inf"), "summary": {"error": str(exc)}})
                continue

            results.append({
                "params": asdict(params) if is_dataclass(params) else combo,
                "score": float(score),
                "summary": {
                    "total_return": result.portfolio_total_return,
                    "sharpe": result.portfolio_sharpe,
                    "max_drawdown": result.portfolio_max_drawdown,
                    "final_equity": result.final_equity,
                },
            })
            if best is None or score > best[0]:
                best = (score, params, result)

        if best is None:
            raise RuntimeError("multi-symbol grid search produced no successful runs")
        return best[1], best[2], results
