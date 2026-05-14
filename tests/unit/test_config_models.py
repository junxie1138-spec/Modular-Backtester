from __future__ import annotations

from backtester.config.models import (
    DataConfig,
    ExecutionConfig,
    OptimizationConfig,
    PortfolioConfig,
    RunConfig,
    WFOConfig,
)


def test_data_config_required_fields():
    d = DataConfig(symbols=["SPY"], timeframe="1d", start="2020-01-01", end="2024-01-01")
    assert d.source == "csv"


def test_execution_defaults():
    e = ExecutionConfig()
    assert e.initial_cash == 100_000.0
    assert e.commission_bps == 1.0
    assert e.slippage_bps == 2.0
    assert e.allow_fractional is False


def test_portfolio_defaults():
    p = PortfolioConfig()
    assert p.sizing_mode == "percent_equity"
    assert p.size == 1.0


def test_optimization_default_empty_space():
    o = OptimizationConfig()
    assert o.objective == "sharpe"
    assert o.param_space == {}


def test_wfo_defaults():
    w = WFOConfig()
    assert w.enabled is False
    assert w.train_bars is None


def test_run_config_composition():
    rc = RunConfig(
        run_name="x",
        strategy="sma_cross",
        strategy_params={"fast": 10, "slow": 50},
        data=DataConfig(symbols=["SPY"], timeframe="1d", start="2020-01-01", end="2024-01-01"),
        execution=ExecutionConfig(),
        portfolio=PortfolioConfig(),
    )
    assert rc.optimization is None
    assert rc.wfo is None


def test_execution_config_allow_short_defaults_false():
    cfg = ExecutionConfig()
    assert cfg.allow_short is False


def test_execution_config_allow_short_override():
    cfg = ExecutionConfig(allow_short=True)
    assert cfg.allow_short is True


def test_execution_config_trailing_stop_pct_defaults_none():
    from backtester.config.models import ExecutionConfig
    cfg = ExecutionConfig()
    assert cfg.trailing_stop_pct is None


def test_execution_config_trailing_stop_atr_mult_defaults_none():
    from backtester.config.models import ExecutionConfig
    cfg = ExecutionConfig()
    assert cfg.trailing_stop_atr_mult is None


def test_execution_config_trailing_stop_atr_period_defaults_14():
    from backtester.config.models import ExecutionConfig
    cfg = ExecutionConfig()
    assert cfg.trailing_stop_atr_period == 14
