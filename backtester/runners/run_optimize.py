from __future__ import annotations

import argparse
import os
from dataclasses import asdict, is_dataclass
from pathlib import Path

from backtester.config.loader import load_run_config
from backtester.config.validation import validate_run_config
from backtester.core.exceptions import ConfigError
from backtester.data.loader import load_symbol
from backtester.data.validators import validate_ohlcv
from backtester.engine.backtest_engine import BacktestEngine
from backtester.engine.broker import Broker
from backtester.engine.multi_backtest_engine import MultiSymbolBacktestEngine
from backtester.engine.multi_portfolio import MultiSymbolPortfolioSimulator
from backtester.engine.portfolio import PortfolioSimulator
from backtester.io.artifacts import ArtifactWriter
from backtester.io.logging import configure_logging
from backtester.io.serialization import write_json
from backtester.optimize.grid_search import GridSearchOptimizer
from backtester.optimize.multi_grid_search import MultiSymbolGridSearchOptimizer
from backtester.strategies.registry import get_strategy_class


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser("run_optimize")
    parser.add_argument("--config", required=True, type=Path)
    args = parser.parse_args(argv)

    rc = load_run_config(args.config)
    validate_run_config(rc)
    if rc.optimization is None:
        raise ConfigError("run_optimize requires an `optimization` block in the config")

    output_root = os.environ.get("BACKTESTER_OUTPUT_ROOT", rc.output_root)
    writer = ArtifactWriter(root=output_root, run_name=rc.run_name)
    log = configure_logging(writer.log_path())

    cls = get_strategy_class(rc.strategy)

    if getattr(cls, "uses_multi_symbol", False):
        return _run_multi_symbol(rc=rc, cls=cls, writer=writer, log=log)
    return _run_legacy_single_symbol(rc=rc, cls=cls, writer=writer, log=log)


def _run_legacy_single_symbol(*, rc, cls, writer, log) -> int:
    if len(rc.data.symbols) != 1:
        raise SystemExit("run_optimize (legacy path) expects exactly one symbol")

    symbol = rc.data.symbols[0]
    data = load_symbol(symbol=symbol, source=rc.data.source, root=rc.data.root,
                       start=rc.data.start, end=rc.data.end)
    validate_ohlcv(data)

    broker = Broker(rc.execution)
    portfolio = PortfolioSimulator(rc.portfolio, initial_cash=rc.execution.initial_cash)
    engine = BacktestEngine(broker=broker, portfolio=portfolio)
    optimizer = GridSearchOptimizer(engine=engine, objective=rc.optimization.objective)

    log.info("grid search: strategy=%s space=%s objective=%s",
             rc.strategy, rc.optimization.param_space, rc.optimization.objective)
    best_params, best_result, all_results = optimizer.find_best(
        strategy_cls=cls, data=data, param_space=rc.optimization.param_space,
        symbol=symbol, timeframe=rc.data.timeframe,
    )

    writer.write_config(rc)
    writer.write_result(best_result)
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


def _run_multi_symbol(*, rc, cls, writer, log) -> int:
    from backtester.config.universe import load_universe_config

    if rc.universe_path is None:
        raise SystemExit("multi-symbol optimize requires universe_path in run config")

    universe = load_universe_config(
        path=Path(rc.universe_path), global_params=rc.strategy_params,
    )
    symbols = list(universe.keys())
    sectors = {sym: cfg.sector for sym, cfg in universe.items()}

    data: dict = {}
    for sym in symbols:
        d = load_symbol(symbol=sym, source=rc.data.source, root=rc.data.root,
                        start=rc.data.start, end=rc.data.end,
                        auto_adjust=getattr(rc.data, "auto_adjust", True))
        validate_ohlcv(d)
        data[sym] = d

    aux_data: dict = {}
    for aux_sym in rc.data.aux_symbols:
        a = load_symbol(symbol=aux_sym, source=rc.data.source, root=rc.data.root,
                        start=rc.data.start, end=rc.data.end,
                        auto_adjust=getattr(rc.data, "auto_adjust", True),
                        require_volume=False)
        validate_ohlcv(a, strict_volume=False)
        aux_data[aux_sym] = a

    sim = MultiSymbolPortfolioSimulator(
        config=rc.portfolio, initial_cash=rc.execution.initial_cash,
        broker_factory=lambda: Broker(rc.execution),
        timeframe=rc.data.timeframe,
    )
    engine = MultiSymbolBacktestEngine(simulator=sim)
    optimizer = MultiSymbolGridSearchOptimizer(
        engine=engine, objective=rc.optimization.objective,
    )

    sampling = rc.optimization.sampling
    random_n = rc.optimization.random_n
    random_seed = rc.optimization.random_seed
    log.info("multi-symbol %s sweep: strategy=%s objective=%s sampling=%s random_n=%d",
             sampling, rc.strategy, rc.optimization.objective, sampling, random_n)

    best_params, best_result, all_results = optimizer.find_best(
        strategy_cls=cls, symbols=symbols, data=data, sectors=sectors,
        aux_data=aux_data, param_space=rc.optimization.param_space,
        regime_config=rc.regimes, sampling=sampling,
        random_n=random_n, random_seed=random_seed,
    )

    writer.write_config(rc, resolved_universe=universe)
    best_result.equity_curve.to_csv(writer.run_dir / "portfolio_equity_curve.csv", index_label="timestamp")
    write_json(writer.run_dir / "summary.json", {
        "best_params": asdict(best_params) if is_dataclass(best_params) else dict(best_params),
        "best_score_objective": rc.optimization.objective,
        "best_summary": {
            "portfolio_total_return": best_result.portfolio_total_return,
            "portfolio_max_drawdown": best_result.portfolio_max_drawdown,
            "portfolio_sharpe": best_result.portfolio_sharpe,
            "final_equity": best_result.final_equity,
        },
        "n_combos": len(all_results),
    })
    write_json(writer.run_dir / "grid_results.json", all_results)
    log.info("multi-symbol optimize complete: best=%s score=%.4f",
             best_params, max((r["score"] for r in all_results if r["score"] != float("-inf")), default=0.0))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
