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


def test_data_config_auto_adjust_defaults_true():
    from backtester.config.models import DataConfig
    cfg = DataConfig(symbols=["SPY"], timeframe="1d", start="2024-01-01", end="2024-12-31")
    assert cfg.auto_adjust is True


def test_data_config_aux_symbols_defaults_empty():
    from backtester.config.models import DataConfig
    cfg = DataConfig(symbols=["SPY"], timeframe="1d", start="2024-01-01", end="2024-12-31")
    assert cfg.aux_symbols == []


def test_data_config_accepts_yfinance_source():
    from backtester.config.models import DataConfig
    cfg = DataConfig(symbols=["SPY"], timeframe="1d", start="2024-01-01", end="2024-12-31", source="yfinance")
    assert cfg.source == "yfinance"


def test_execution_config_hard_stop_atr_mult_defaults_none():
    from backtester.config.models import ExecutionConfig
    cfg = ExecutionConfig()
    assert cfg.hard_stop_atr_mult is None


def test_execution_config_runner_atr_mult_defaults_none():
    from backtester.config.models import ExecutionConfig
    cfg = ExecutionConfig()
    assert cfg.runner_atr_mult is None


def test_execution_config_breakeven_floor_defaults_true():
    from backtester.config.models import ExecutionConfig
    cfg = ExecutionConfig()
    assert cfg.breakeven_floor is True


def test_execution_config_tranche_stop_atr_period_defaults_20():
    from backtester.config.models import ExecutionConfig
    cfg = ExecutionConfig()
    assert cfg.tranche_stop_atr_period == 20


def test_portfolio_config_sizing_mode_default_percent_equity():
    from backtester.config.models import PortfolioConfig
    cfg = PortfolioConfig()
    assert cfg.sizing_mode == "percent_equity"


def test_portfolio_config_vol_target_defaults_012():
    from backtester.config.models import PortfolioConfig
    cfg = PortfolioConfig()
    assert cfg.vol_target == 0.12


def test_portfolio_config_position_cap_pct_defaults_1():
    from backtester.config.models import PortfolioConfig
    cfg = PortfolioConfig()
    assert cfg.position_cap_pct == 1.0


def test_portfolio_config_cash_reserve_pct_defaults_0():
    from backtester.config.models import PortfolioConfig
    cfg = PortfolioConfig()
    assert cfg.cash_reserve_pct == 0.0


def test_portfolio_config_risk_budget_pct_defaults_1():
    from backtester.config.models import PortfolioConfig
    cfg = PortfolioConfig()
    assert cfg.risk_budget_pct == 1.0


def test_portfolio_config_sector_cap_pct_defaults_1():
    from backtester.config.models import PortfolioConfig
    cfg = PortfolioConfig()
    assert cfg.sector_cap_pct == 1.0


def test_spy_ema_regime_config_defaults():
    from backtester.config.models import SpyEmaRegimeConfig
    cfg = SpyEmaRegimeConfig()
    assert cfg.enabled is False
    assert cfg.ema_lookback == 200
    assert cfg.trip_pct == -0.02
    assert cfg.resume_pct == 0.02


def test_vix_regime_config_defaults():
    from backtester.config.models import VixRegimeConfig
    cfg = VixRegimeConfig()
    assert cfg.enabled is False
    assert cfg.trip_threshold == 30.0
    assert cfg.trip_consec == 2
    assert cfg.resume_threshold == 25.0
    assert cfg.resume_consec == 3


def test_circuit_breaker_config_defaults():
    from backtester.config.models import CircuitBreakerConfig
    cfg = CircuitBreakerConfig()
    assert cfg.enabled is False
    assert cfg.pnl_window_days == 20
    assert cfg.trip_pct == -0.05
    assert cfg.pause_days == 10


def test_regimes_config_holds_three_subconfigs():
    from backtester.config.models import RegimesConfig
    cfg = RegimesConfig()
    assert cfg.spy_ema.enabled is False
    assert cfg.vix.enabled is False
    assert cfg.circuit_breaker.enabled is False


def test_run_config_universe_path_defaults_none():
    from backtester.config.models import RunConfig
    import dataclasses
    fields = {f.name: f for f in dataclasses.fields(RunConfig)}
    assert "universe_path" in fields
    assert fields["universe_path"].default is None


def test_run_config_regimes_defaults_none():
    from backtester.config.models import RunConfig
    import dataclasses
    fields = {f.name: f for f in dataclasses.fields(RunConfig)}
    assert "regimes" in fields
    assert fields["regimes"].default is None
