from __future__ import annotations

import argparse
from pathlib import Path

from backtester.config.loader import load_run_config
from backtester.config.validation import validate_run_config
from backtester.core.exceptions import ConfigError
from backtester.data.loader import load_symbol
from backtester.data.validators import validate_ohlcv
from backtester.engine.backtest_engine import BacktestEngine
from backtester.engine.broker import Broker
from backtester.engine.portfolio import PortfolioSimulator
from backtester.io.artifacts import ArtifactWriter
from backtester.io.logging import configure_logging
from backtester.io.serialization import write_json
from backtester.optimize.grid_search import GridSearchOptimizer
from backtester.strategies.registry import get_strategy_class


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser("run_optimize")
    parser.add_argument("--config", required=True, type=Path)
    args = parser.parse_args(argv)

    rc = load_run_config(args.config)
    validate_run_config(rc)
    if rc.optimization is None:
        raise ConfigError("run_optimize requires an `optimization` block in the config")
    if len(rc.data.symbols) != 1:
        raise SystemExit("run_optimize expects exactly one symbol")

    writer = ArtifactWriter(root=rc.output_root, run_name=rc.run_name)
    log = configure_logging(writer.log_path())

    symbol = rc.data.symbols[0]
    data = load_symbol(symbol=symbol, source=rc.data.source, root=rc.data.root,
                       start=rc.data.start, end=rc.data.end)
    validate_ohlcv(data)

    cls = get_strategy_class(rc.strategy)
    broker = Broker(rc.execution)
    portfolio = PortfolioSimulator(rc.portfolio, initial_cash=rc.execution.initial_cash)
    engine = BacktestEngine(broker=broker, portfolio=portfolio)
    optimizer = GridSearchOptimizer(engine=engine, objective=rc.optimization.objective)

    log.info("grid search: strategy=%s space=%s objective=%s",
             rc.strategy, rc.optimization.param_space, rc.optimization.objective)
    best_params, best_result, all_results = optimizer.find_best(
        strategy_cls=cls,
        data=data,
        param_space=rc.optimization.param_space,
        symbol=symbol,
        timeframe=rc.data.timeframe,
    )

    writer.write_config(rc)
    writer.write_result(best_result)
    # Overwrite summary.json with optimizer-aware payload
    write_json(writer.run_dir / "summary.json", {
        "best_params": best_result.summary["params"],
        "best_score_objective": rc.optimization.objective,
        "best_summary": best_result.summary,
    })
    write_json(writer.run_dir / "grid_results.json", all_results)
    log.info("best=%s score(%s)=%.4f",
             best_result.summary["params"], rc.optimization.objective,
             max(r["score"] for r in all_results))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
