from __future__ import annotations

import argparse
import logging
from pathlib import Path

from backtester.config.loader import load_run_config
from backtester.config.validation import validate_run_config
from backtester.data.loader import load_symbol
from backtester.data.validators import validate_ohlcv
from backtester.engine.backtest_engine import BacktestEngine
from backtester.engine.broker import Broker
from backtester.engine.portfolio import PortfolioSimulator
from backtester.io.artifacts import ArtifactWriter
from backtester.io.logging import configure_logging
from backtester.strategies.instantiate import build_strategy_and_params


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser("run_backtest")
    parser.add_argument("--config", required=True, type=Path)
    args = parser.parse_args(argv)

    rc = load_run_config(args.config)
    validate_run_config(rc)

    if len(rc.data.symbols) != 1:
        raise SystemExit("run_backtest expects exactly one symbol; use run_batch for many")

    writer = ArtifactWriter(root=rc.output_root, run_name=rc.run_name)
    log = configure_logging(writer.log_path())
    log.info("run_dir=%s", writer.run_dir)

    symbol = rc.data.symbols[0]
    data = load_symbol(symbol=symbol, source=rc.data.source, root=rc.data.root,
                       start=rc.data.start, end=rc.data.end)
    validate_ohlcv(data)

    strategy, params = build_strategy_and_params(rc.strategy, rc.strategy_params)
    broker = Broker(rc.execution)
    portfolio = PortfolioSimulator(rc.portfolio, initial_cash=rc.execution.initial_cash)
    engine = BacktestEngine(broker=broker, portfolio=portfolio)

    log.info("running %s on %s (%d bars)", rc.strategy, symbol, len(data))
    result = engine.run(strategy, data, params, symbol=symbol, timeframe=rc.data.timeframe)
    log.info("done: total_return=%.4f sharpe=%.3f max_dd=%.3f",
             result.summary["total_return"], result.summary["sharpe"], result.summary["max_drawdown"])

    writer.write_config(rc)
    writer.write_result(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
