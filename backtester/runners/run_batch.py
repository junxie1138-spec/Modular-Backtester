from __future__ import annotations

import argparse
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
from backtester.io.serialization import write_json
from backtester.strategies.instantiate import build_strategy_and_params


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser("run_batch")
    parser.add_argument("--config", required=True, type=Path)
    args = parser.parse_args(argv)

    rc = load_run_config(args.config)
    validate_run_config(rc)
    if not rc.data.symbols:
        raise SystemExit("data.symbols is empty")

    writer = ArtifactWriter(root=rc.output_root, run_name=rc.run_name)
    log = configure_logging(writer.log_path())

    by_symbol = {}
    for symbol in rc.data.symbols:
        try:
            data = load_symbol(symbol=symbol, source=rc.data.source, root=rc.data.root,
                               start=rc.data.start, end=rc.data.end)
            validate_ohlcv(data)
            strategy, params = build_strategy_and_params(rc.strategy, rc.strategy_params)
            broker = Broker(rc.execution)
            portfolio = PortfolioSimulator(rc.portfolio, initial_cash=rc.execution.initial_cash)
            engine = BacktestEngine(broker=broker, portfolio=portfolio)
            result = engine.run(strategy, data, params, symbol=symbol, timeframe=rc.data.timeframe)
            by_symbol[symbol] = result.summary
            result.equity_curve.to_csv(writer.run_dir / f"{symbol}_equity_curve.csv", index_label="timestamp")
            result.trades.to_csv(writer.run_dir / f"{symbol}_trades.csv", index=False)
            log.info("%s: total_return=%.4f sharpe=%.3f", symbol,
                     result.summary["total_return"], result.summary["sharpe"])
        except Exception as exc:
            log.warning("%s failed: %s", symbol, exc)
            by_symbol[symbol] = {"error": str(exc)}

    writer.write_config(rc)
    write_json(writer.run_dir / "batch_summary.json", by_symbol)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
