from __future__ import annotations

from dataclasses import asdict, is_dataclass
import logging
from typing import Any, Dict, Type

import pandas as pd

from backtester.config.models import RunConfig
from backtester.engine.backtest_engine import BacktestEngine
from backtester.optimize.grid_search import GridSearchOptimizer
from backtester.strategies.base import BaseStrategy
from backtester.wfo.splitter import WalkForwardSplitter
from backtester.wfo.stitcher import WalkForwardStitcher

log = logging.getLogger("backtester.wfo")


class WalkForwardRunner:
    def __init__(
        self,
        engine: BacktestEngine,
        optimizer: GridSearchOptimizer,
        splitter: WalkForwardSplitter,
        stitcher: WalkForwardStitcher,
    ):
        self.engine = engine
        self.optimizer = optimizer
        self.splitter = splitter
        self.stitcher = stitcher

    def run(
        self,
        strategy_cls: Type[BaseStrategy],
        full_data: pd.DataFrame,
        base_config: RunConfig,
    ) -> Dict[str, Any]:
        assert base_config.wfo is not None and base_config.wfo.enabled
        assert base_config.optimization is not None

        windows = self.splitter.split(
            data=full_data,
            train_bars=base_config.wfo.train_bars,
            test_bars=base_config.wfo.test_bars,
            step_bars=base_config.wfo.step_bars,
        )
        log.info("WFO: %d windows", len(windows))

        symbol = base_config.data.symbols[0]
        timeframe = base_config.data.timeframe

        window_results = []
        for i, window in enumerate(windows):
            best_params, _train_result, _all_train = self.optimizer.find_best(
                strategy_cls=strategy_cls,
                data=window.train_data,
                param_space=base_config.optimization.param_space,
                symbol=symbol,
                timeframe=timeframe,
            )

            test_strategy = strategy_cls()
            test_result = self.engine.run(
                strategy=test_strategy,
                data=window.test_data,
                params=best_params,
                symbol=symbol,
                timeframe=timeframe,
            )

            window_results.append({
                "window_index": i,
                "train_start": window.train_data.index.min(),
                "train_end": window.train_data.index.max(),
                "test_start": window.test_data.index.min(),
                "test_end": window.test_data.index.max(),
                "best_params": asdict(best_params) if is_dataclass(best_params) else dict(best_params),
                "train_summary": _train_result.summary,
                "test_summary": test_result.summary,
                "test_result": test_result,
            })
            log.info("window %d: best=%s test_sharpe=%.3f",
                     i, window_results[-1]["best_params"], test_result.summary.get("sharpe", 0.0))

        return self.stitcher.combine(window_results)
