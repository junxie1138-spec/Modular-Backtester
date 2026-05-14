import pandas as pd
import pytest


def _ohlcv(closes, start="2024-01-02"):
    idx = pd.date_range(start, periods=len(closes), freq="B")
    return pd.DataFrame(
        {"open": closes, "high": [c+0.5 for c in closes], "low": [c-0.5 for c in closes],
         "close": closes, "volume": [1_000_000]*len(closes)},
        index=idx,
    )


class _MinimalStrategy:
    """Minimal multi-symbol strategy stub for optimizer testing."""
    from dataclasses import dataclass

    @dataclass(slots=True)
    class _Params:
        threshold: float = 0.0
        size: float = 1.0

    strategy_id = "test_opt_stub"
    uses_multi_symbol = True
    uses_per_bar = False

    @classmethod
    def params_type(cls):
        return cls._Params

    def indicators(self, data, params):
        return pd.DataFrame({"sma5": data["close"].rolling(5).mean()}, index=data.index)

    def generate_signals_for_symbol(self, *, data, indicators, params):
        sig = (data["close"] > indicators["sma5"] + params.threshold).astype(float).shift(1).fillna(0)
        return pd.DataFrame({"signal": sig, "size": params.size}, index=data.index)


def test_optimizer_runs_grid_mode():
    from backtester.optimize.multi_grid_search import MultiSymbolGridSearchOptimizer
    from backtester.config.models import PortfolioConfig, ExecutionConfig
    from backtester.engine.broker import Broker
    from backtester.engine.multi_portfolio import MultiSymbolPortfolioSimulator
    from backtester.engine.multi_backtest_engine import MultiSymbolBacktestEngine

    sim = MultiSymbolPortfolioSimulator(
        config=PortfolioConfig(sizing_mode="percent_equity", size=0.1,
                               position_cap_pct=1.0, cash_reserve_pct=0.0,
                               risk_budget_pct=1.0, sector_cap_pct=1.0),
        initial_cash=100_000.0,
        broker_factory=lambda: Broker(ExecutionConfig(initial_cash=100_000.0)),
    )
    engine = MultiSymbolBacktestEngine(simulator=sim)
    optimizer = MultiSymbolGridSearchOptimizer(engine=engine, objective="sharpe")

    closes = [100.0 + i*0.5 - (i%5)*1.0 for i in range(30)]
    best_params, best_result, all_results = optimizer.find_best(
        strategy_cls=_MinimalStrategy,
        symbols=["AAA"], data={"AAA": _ohlcv(closes)}, sectors={"AAA": "X"},
        aux_data={},
        param_space={"threshold": [0.0, 0.5, 1.0]},
        sampling="grid",
    )
    assert len(all_results) == 3
    assert hasattr(best_result, "equity_curve")


def test_optimizer_runs_lhs_mode():
    from backtester.optimize.multi_grid_search import MultiSymbolGridSearchOptimizer
    from backtester.config.models import PortfolioConfig, ExecutionConfig
    from backtester.engine.broker import Broker
    from backtester.engine.multi_portfolio import MultiSymbolPortfolioSimulator
    from backtester.engine.multi_backtest_engine import MultiSymbolBacktestEngine

    sim = MultiSymbolPortfolioSimulator(
        config=PortfolioConfig(sizing_mode="percent_equity", size=0.1,
                               position_cap_pct=1.0, cash_reserve_pct=0.0,
                               risk_budget_pct=1.0, sector_cap_pct=1.0),
        initial_cash=100_000.0,
        broker_factory=lambda: Broker(ExecutionConfig(initial_cash=100_000.0)),
    )
    engine = MultiSymbolBacktestEngine(simulator=sim)
    optimizer = MultiSymbolGridSearchOptimizer(engine=engine, objective="total_return")

    closes = [100.0 + i*0.3 for i in range(30)]
    best_params, best_result, all_results = optimizer.find_best(
        strategy_cls=_MinimalStrategy,
        symbols=["AAA"], data={"AAA": _ohlcv(closes)}, sectors={"AAA": "X"},
        aux_data={},
        param_space={"threshold": [0.0, 0.5, 1.0, 1.5, 2.0]},
        sampling="lhs", random_n=4, random_seed=0,
    )
    assert len(all_results) == 4  # random_n samples
