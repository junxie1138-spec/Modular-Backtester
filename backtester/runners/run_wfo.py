from __future__ import annotations

import argparse
import os
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
from backtester.wfo.runner import WalkForwardRunner
from backtester.wfo.splitter import WalkForwardSplitter
from backtester.wfo.stitcher import WalkForwardStitcher
from backtester.wfo.multi_runner import MultiSymbolWFORunner
from backtester.wfo.multi_splitter import MultiSymbolWFOSplitter
from backtester.wfo.multi_stitcher import MultiSymbolWFOStitcher


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

    cls = get_strategy_class(rc.strategy)
    is_multi = getattr(cls, "uses_multi_symbol", False)

    if is_multi:
        if rc.universe_path is None:
            raise SystemExit("multi-symbol WFO requires universe_path in run config")
        return _run_multi_symbol_wfo(rc=rc, cls=cls)
    if len(rc.data.symbols) != 1:
        raise SystemExit("run_wfo (single-symbol path) expects exactly one symbol")
    return _run_legacy_single_symbol_wfo(rc=rc, cls=cls)


def _run_legacy_single_symbol_wfo(*, rc, cls) -> int:
    output_root = os.environ.get("BACKTESTER_OUTPUT_ROOT", rc.output_root)
    writer = ArtifactWriter(root=output_root, run_name=rc.run_name)
    log = configure_logging(writer.log_path())

    symbol = rc.data.symbols[0]
    data = load_symbol(symbol=symbol, source=rc.data.source, root=rc.data.root,
                       start=rc.data.start, end=rc.data.end)
    validate_ohlcv(data)

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


def _run_multi_symbol_wfo(*, rc, cls) -> int:
    from backtester.config.universe import load_universe_config

    output_root = os.environ.get("BACKTESTER_OUTPUT_ROOT", rc.output_root)
    writer = ArtifactWriter(root=output_root, run_name=rc.run_name)
    log = configure_logging(writer.log_path())

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

    # Align all symbol panels to the union of indices BEFORE the splitter.
    # IPO-late symbols (COIN, PLTR, XPEV, NIO) have shorter histories and would
    # otherwise produce misaligned per-symbol slices in the splitter.
    from backtester.engine.multi_backtest_engine import _align_panel
    data, aux_data = _align_panel(data, aux_data)

    sim = MultiSymbolPortfolioSimulator(
        config=rc.portfolio, initial_cash=rc.execution.initial_cash,
        broker_factory=lambda: Broker(rc.execution),
        timeframe=rc.data.timeframe,
    )
    engine = MultiSymbolBacktestEngine(simulator=sim)
    optimizer = MultiSymbolGridSearchOptimizer(
        engine=engine, objective=rc.optimization.objective,
    )
    splitter = MultiSymbolWFOSplitter(
        train_bars=rc.wfo.train_bars, test_bars=rc.wfo.test_bars,
        step_bars=rc.wfo.step_bars,
    )
    runner = MultiSymbolWFORunner(engine=engine, optimizer=optimizer)
    stitcher = MultiSymbolWFOStitcher(timeframe=rc.data.timeframe)

    log.info("running multi-symbol WFO: train=%d test=%d step=%d on %d symbols",
             rc.wfo.train_bars, rc.wfo.test_bars, rc.wfo.step_bars, len(symbols))

    sampling = rc.optimization.sampling
    random_n = rc.optimization.random_n
    random_seed = rc.optimization.random_seed

    window_results = []
    for window in splitter.split(data=data, aux_data=aux_data):
        log.info("window %d: train [%s, %s] test [%s, %s]",
                 window.window_idx, window.train_start.date(), window.train_end.date(),
                 window.test_start.date(), window.test_end.date())
        wr = runner.run_window(
            strategy_cls=cls, symbols=symbols, sectors=sectors,
            window=window, param_space=rc.optimization.param_space,
            regime_config=rc.regimes, sampling=sampling,
            random_n=random_n, random_seed=random_seed,
        )
        window_results.append(wr)

    if not window_results:
        raise SystemExit("no WFO windows produced — check train_bars/test_bars vs data length")

    stitched = stitcher.stitch(window_results)

    writer.write_config(rc, resolved_universe=universe)
    stitched.oos_equity_curve.to_csv(writer.run_dir / "oos_equity_curve.csv", index_label="timestamp")
    write_json(writer.run_dir / "summary.json", {
        "oos_summary": stitched.oos_summary,
        "parameter_stability": stitched.parameter_stability,
        "n_windows": len(window_results),
    })
    write_json(writer.run_dir / "window_results.json", stitched.window_summaries)
    log.info("multi-symbol WFO complete: %d windows, oos_sharpe=%.3f, oos_total_return=%.4f",
             len(window_results), stitched.oos_summary.get("sharpe", 0.0),
             stitched.oos_summary.get("total_return", 0.0))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
