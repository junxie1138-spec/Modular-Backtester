from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

import pandas as pd

from backtester.engine.multi_backtest_engine import MultiSymbolBacktestEngine
from backtester.optimize.multi_grid_search import MultiSymbolGridSearchOptimizer
from backtester.strategies.base import BaseStrategy
from backtester.wfo.multi_splitter import WindowPanel


@dataclass
class WindowResult:
    """Per-window OOS result."""
    window_idx: int
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp
    best_params: Any
    is_summary: dict[str, float]
    oos_summary: dict[str, float]
    oos_equity_curve: pd.Series


@dataclass
class MultiSymbolWFORunner:
    """Runs walk-forward optimization on a multi-symbol panel.

    For each window:
      1. Train: optimize over the param_space using MultiSymbolGridSearchOptimizer.
      2. Test: run MultiSymbolBacktestEngine with the best params on the test slice.
      3. Collect IS and OOS summaries plus the OOS equity curve.
    """
    engine: MultiSymbolBacktestEngine
    optimizer: MultiSymbolGridSearchOptimizer

    def run_window(
        self,
        *,
        strategy_cls: type[BaseStrategy],
        symbols: list[str],
        sectors: dict[str, str],
        window: WindowPanel,
        param_space: dict[str, list[Any]],
        regime_config: Optional[Any] = None,
        sampling: str = "grid",
        random_n: int = 100,
        random_seed: int = 0,
    ) -> WindowResult:
        # In-sample optimization.
        best_params, _is_result, _ = self.optimizer.find_best(
            strategy_cls=strategy_cls,
            symbols=symbols, data=window.train_data, sectors=sectors,
            aux_data=window.train_aux,
            param_space=param_space, regime_config=regime_config,
            sampling=sampling, random_n=random_n, random_seed=random_seed,
        )
        is_summary = {
            "portfolio_total_return": _is_result.portfolio_total_return,
            "portfolio_max_drawdown": _is_result.portfolio_max_drawdown,
            "portfolio_sharpe": _is_result.portfolio_sharpe,
        }

        # Out-of-sample run with best params.
        strategy = strategy_cls()
        oos_result = self.engine.run(
            strategy=strategy, symbols=symbols, data=window.test_data,
            sectors=sectors, aux_data=window.test_aux,
            params=best_params, regime_config=regime_config,
        )
        oos_summary = {
            "portfolio_total_return": oos_result.portfolio_total_return,
            "portfolio_max_drawdown": oos_result.portfolio_max_drawdown,
            "portfolio_sharpe": oos_result.portfolio_sharpe,
            "final_equity": oos_result.final_equity,
        }

        return WindowResult(
            window_idx=window.window_idx,
            train_start=window.train_start, train_end=window.train_end,
            test_start=window.test_start, test_end=window.test_end,
            best_params=best_params,
            is_summary=is_summary,
            oos_summary=oos_summary,
            oos_equity_curve=oos_result.equity_curve,
        )
