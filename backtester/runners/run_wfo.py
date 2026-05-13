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
from backtester.wfo.runner import WalkForwardRunner
from backtester.wfo.splitter import WalkForwardSplitter
from backtester.wfo.stitcher import WalkForwardStitcher


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser("run_wfo")
    parser.add_argument("--config", required=True, type=Path)
    args = parser.parse_args(argv)

    rc = load_run_config(args.config)
    validate_run_config(rc)
    if rc.wfo is None or not rc.wfo.enabled:
        raise ConfigError("run_wfo requires `wfo.enabled: true`")
    if rc.optimization is None:
        raise ConfigError("run_wfo requires an `optimization` block")
    if len(rc.data.symbols) != 1:
        raise SystemExit("run_wfo expects exactly one symbol")

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
    runner = WalkForwardRunner(engine=engine, optimizer=optimizer,
                               splitter=WalkForwardSplitter(), stitcher=WalkForwardStitcher())

    log.info("running WFO: train=%d test=%d step=%d",
             rc.wfo.train_bars, rc.wfo.test_bars, rc.wfo.step_bars)
    out = runner.run(strategy_cls=cls, full_data=data, base_config=rc)

    writer.write_config(rc)
    write_json(writer.run_dir / "summary.json", {
        "oos_summary": out["oos_summary"],
        "is_summary_avg": out["is_summary_avg"],
        "parameter_stability": out["parameter_stability"],
        "n_windows": len(out["window_results"]),
    })
    write_json(writer.run_dir / "window_results.json", out["window_results"])
    out["oos_equity_curve"].to_csv(writer.run_dir / "oos_equity_curve.csv", index_label="timestamp")
    out["oos_trades"].to_csv(writer.run_dir / "oos_trades.csv", index=False)
    out["oos_positions"].to_csv(writer.run_dir / "oos_positions.csv", index_label="timestamp")

    log.info("WFO complete: %d windows, oos_sharpe=%.3f",
             len(out["window_results"]), out["oos_summary"].get("sharpe", 0.0))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
