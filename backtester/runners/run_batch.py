from __future__ import annotations

import argparse
import os
from pathlib import Path

from backtester.config.loader import load_run_config
from backtester.config.validation import validate_run_config
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
from backtester.strategies.instantiate import build_strategy_and_params


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser("run_batch")
    parser.add_argument("--config", required=True, type=Path)
    args = parser.parse_args(argv)

    rc = load_run_config(args.config)
    validate_run_config(rc)

    output_root = os.environ.get("BACKTESTER_OUTPUT_ROOT", rc.output_root)
    writer = ArtifactWriter(root=output_root, run_name=rc.run_name)
    log = configure_logging(writer.log_path())

    strategy, params = build_strategy_and_params(rc.strategy, rc.strategy_params)

    if getattr(strategy, "uses_multi_symbol", False):
        return _run_multi_symbol(rc=rc, strategy=strategy, params=params,
                                  writer=writer, log=log)
    return _run_legacy_per_symbol(rc=rc, strategy=strategy, params=params,
                                   writer=writer, log=log)


def _run_legacy_per_symbol(*, rc, strategy, params, writer, log) -> int:
    """v0.3.0 path: independent per-symbol runs, no shared cash."""
    if not rc.data.symbols:
        raise SystemExit("data.symbols is empty (and strategy is not multi-symbol)")
    by_symbol = {}
    for symbol in rc.data.symbols:
        try:
            data = load_symbol(symbol=symbol, source=rc.data.source, root=rc.data.root,
                               start=rc.data.start, end=rc.data.end)
            validate_ohlcv(data)
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


def _run_multi_symbol(*, rc, strategy, params, writer, log) -> int:
    """v0.4.0 path: single shared-cash run across the universe."""
    from backtester.config.universe import load_universe_config

    if rc.universe_path is None:
        raise SystemExit("multi-symbol strategy requires universe_path in run config")

    universe = load_universe_config(
        path=Path(rc.universe_path), global_params=rc.strategy_params,
    )
    symbols = list(universe.keys())
    sectors = {sym: cfg.sector for sym, cfg in universe.items()}

    # Load OHLCV for every symbol AND aux_symbols.
    data: dict = {}
    for sym in symbols:
        d = load_symbol(
            symbol=sym, source=rc.data.source, root=rc.data.root,
            start=rc.data.start, end=rc.data.end,
            auto_adjust=getattr(rc.data, "auto_adjust", True),
        )
        validate_ohlcv(d)
        data[sym] = d

    aux_data: dict = {}
    for aux_sym in rc.data.aux_symbols:
        a = load_symbol(
            symbol=aux_sym, source=rc.data.source, root=rc.data.root,
            start=rc.data.start, end=rc.data.end,
            auto_adjust=getattr(rc.data, "auto_adjust", True),
            require_volume=False,
        )
        validate_ohlcv(a, strict_volume=False)
        aux_data[aux_sym] = a

    sim = MultiSymbolPortfolioSimulator(
        config=rc.portfolio,
        initial_cash=rc.execution.initial_cash,
        broker_factory=lambda: Broker(rc.execution),
    )
    engine = MultiSymbolBacktestEngine(simulator=sim)
    result = engine.run(
        strategy=strategy, symbols=symbols, data=data, sectors=sectors,
        aux_data=aux_data, params=params, regime_config=rc.regimes,
    )

    try:
        writer.write_config(rc, resolved_universe=universe)
    except TypeError:
        # write_config doesn't yet accept resolved_universe — Task 37 adds it.
        writer.write_config(rc)

    result.equity_curve.to_csv(writer.run_dir / "portfolio_equity_curve.csv", index_label="timestamp")
    write_json(writer.run_dir / "batch_summary.json", {
        "portfolio_total_return": result.portfolio_total_return,
        "portfolio_max_drawdown": result.portfolio_max_drawdown,
        "portfolio_sharpe": result.portfolio_sharpe,
        "final_equity": result.final_equity,
        "n_symbols": len(symbols),
    })
    log.info("multi-symbol run complete: %d symbols, total_return=%.4f",
             len(symbols), result.portfolio_total_return)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
