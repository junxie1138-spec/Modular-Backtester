from __future__ import annotations

import pytest

from backtester.config.models import (
    DataConfig, ExecutionConfig, PortfolioConfig, RunConfig, WFOConfig,
    RegimesConfig,
)
from backtester.config.validation import validate_run_config
from backtester.core.exceptions import ConfigError


def _base_run_config():
    """Helper: smallest valid RunConfig for v0.4.0 validation tests."""
    return RunConfig(
        run_name="vtest",
        strategy="sma_cross",
        strategy_params={"fast": 10, "slow": 30, "size": 1.0},
        data=DataConfig(source="csv", root="data/raw",
                        start="2024-01-01", end="2024-06-30", timeframe="1d",
                        symbols=["SPY"]),
        execution=ExecutionConfig(),
        portfolio=PortfolioConfig(sizing_mode="percent_equity", size=0.95),
        output_root="output/runs",
    )


def _base_with_regimes():
    """Helper: base config with regimes enabled."""
    rc = _base_run_config()
    rc.regimes = RegimesConfig()
    return rc


def _make(**over):
    base = RunConfig(
        run_name="x",
        strategy="sma_cross",
        strategy_params={"fast": 10, "slow": 30},
        data=DataConfig(symbols=["SPY"], timeframe="1d", start="2020-01-01", end="2024-01-01"),
        execution=ExecutionConfig(),
        portfolio=PortfolioConfig(),
    )
    for k, v in over.items():
        setattr(base, k, v)
    return base


def test_valid_config_passes():
    validate_run_config(_make())


def test_empty_symbols_rejected():
    rc = _make()
    rc.data.symbols = []
    with pytest.raises(ConfigError, match="symbols"):
        validate_run_config(rc)


def test_start_after_end_rejected():
    rc = _make()
    rc.data.start = "2025-01-01"
    rc.data.end = "2024-01-01"
    with pytest.raises(ConfigError, match="start"):
        validate_run_config(rc)


def test_negative_cash_rejected():
    rc = _make()
    rc.execution.initial_cash = -1
    with pytest.raises(ConfigError, match="initial_cash"):
        validate_run_config(rc)


def test_wfo_requires_windows():
    rc = _make(wfo=WFOConfig(enabled=True))
    with pytest.raises(ConfigError, match="train_bars"):
        validate_run_config(rc)


def test_wfo_valid_when_windows_set():
    rc = _make(wfo=WFOConfig(enabled=True, train_bars=252, test_bars=63, step_bars=63))
    validate_run_config(rc)


def test_trailing_stop_pct_and_atr_mutually_exclusive():
    rc = _make()
    rc.execution.trailing_stop_pct = 0.05
    rc.execution.trailing_stop_atr_mult = 2.0
    with pytest.raises(ConfigError, match="mutually exclusive"):
        validate_run_config(rc)


def test_trailing_stop_pct_out_of_range():
    for bad in (0.0, 1.0, -0.1, 1.5):
        rc = _make()
        rc.execution.trailing_stop_pct = bad
        with pytest.raises(ConfigError, match="trailing_stop_pct"):
            validate_run_config(rc)


def test_trailing_stop_atr_mult_must_be_positive():
    for bad in (0.0, -1.0):
        rc = _make()
        rc.execution.trailing_stop_atr_mult = bad
        with pytest.raises(ConfigError, match="trailing_stop_atr_mult"):
            validate_run_config(rc)


def test_trailing_stop_atr_period_too_small():
    rc = _make()
    rc.execution.trailing_stop_atr_mult = 2.0
    rc.execution.trailing_stop_atr_period = 1
    with pytest.raises(ConfigError, match="trailing_stop_atr_period"):
        validate_run_config(rc)


# Task 11: Tranche-stop validation rules (rules 1-4)
def test_validation_rule_01_hard_and_runner_both_or_neither():
    rc = _base_run_config()
    rc.execution.hard_stop_atr_mult = 1.75
    rc.execution.runner_atr_mult = None
    with pytest.raises(ConfigError, match="both-or-neither"):
        validate_run_config(rc)


def test_validation_rule_02_v030_and_v040_keys_mutually_exclusive():
    rc = _base_run_config()
    rc.execution.trailing_stop_pct = 0.05
    rc.execution.hard_stop_atr_mult = 1.75
    rc.execution.runner_atr_mult = 2.5
    with pytest.raises(ConfigError, match="mutually exclusive"):
        validate_run_config(rc)


def test_validation_rule_03_tranche_stop_mults_positive():
    rc = _base_run_config()
    rc.execution.hard_stop_atr_mult = 0.0
    rc.execution.runner_atr_mult = 2.5
    with pytest.raises(ConfigError, match="hard_stop_atr_mult must be > 0"):
        validate_run_config(rc)


def test_validation_rule_04_tranche_stop_atr_period_min_2():
    rc = _base_run_config()
    rc.execution.hard_stop_atr_mult = 1.75
    rc.execution.runner_atr_mult = 2.5
    rc.execution.tranche_stop_atr_period = 1
    with pytest.raises(ConfigError, match="tranche_stop_atr_period must be >= 2"):
        validate_run_config(rc)


# Task 12: Portfolio sizing validation rules (rules 5-9)
def test_validation_rule_05_position_cap_pct_bounds():
    for bad in [0.0, -0.1, 1.5]:
        rc = _base_run_config()
        rc.portfolio.position_cap_pct = bad
        with pytest.raises(ConfigError, match="position_cap_pct"):
            validate_run_config(rc)


def test_validation_rule_06_cash_reserve_pct_bounds():
    for bad in [-0.01, 1.0, 1.5]:
        rc = _base_run_config()
        rc.portfolio.cash_reserve_pct = bad
        with pytest.raises(ConfigError, match="cash_reserve_pct"):
            validate_run_config(rc)


def test_validation_rule_07_risk_budget_pct_bounds():
    for bad in [0.0, -0.1, 1.5]:
        rc = _base_run_config()
        rc.portfolio.risk_budget_pct = bad
        with pytest.raises(ConfigError, match="risk_budget_pct"):
            validate_run_config(rc)


def test_validation_rule_08_sector_cap_pct_bounds():
    for bad in [0.0, -0.1, 1.5]:
        rc = _base_run_config()
        rc.portfolio.sector_cap_pct = bad
        with pytest.raises(ConfigError, match="sector_cap_pct"):
            validate_run_config(rc)


def test_validation_rule_09_vol_target_positive_when_vol_targeted():
    rc = _base_run_config()
    rc.portfolio.sizing_mode = "vol_targeted"
    rc.portfolio.vol_target = 0.0
    with pytest.raises(ConfigError, match="vol_target must be > 0"):
        validate_run_config(rc)
    # Other modes: vol_target=0 is allowed (field is ignored).
    rc.portfolio.sizing_mode = "percent_equity"
    rc.portfolio.vol_target = 0.0
    validate_run_config(rc)  # no raise


# Task 13: Regime validation rules (rules 10-14)
def test_validation_rule_10_circuit_breaker_pause_days_nonneg():
    rc = _base_with_regimes()
    rc.regimes.circuit_breaker.pause_days = -1
    with pytest.raises(ConfigError, match="pause_days"):
        validate_run_config(rc)


def test_validation_rule_11_vix_resume_below_trip():
    rc = _base_with_regimes()
    rc.regimes.vix.trip_threshold = 25
    rc.regimes.vix.resume_threshold = 30
    with pytest.raises(ConfigError, match="resume_threshold.*trip_threshold"):
        validate_run_config(rc)


def test_validation_rule_12_spy_pct_signs():
    rc = _base_with_regimes()
    rc.regimes.spy_ema.trip_pct = 0.02  # must be <= 0
    with pytest.raises(ConfigError, match="spy_ema.trip_pct"):
        validate_run_config(rc)
    rc = _base_with_regimes()
    rc.regimes.spy_ema.resume_pct = -0.02  # must be >= 0
    with pytest.raises(ConfigError, match="spy_ema.resume_pct"):
        validate_run_config(rc)


def test_validation_rule_13_vix_consec_min_1():
    rc = _base_with_regimes()
    rc.regimes.vix.trip_consec = 0
    with pytest.raises(ConfigError, match="vix.trip_consec"):
        validate_run_config(rc)
    rc = _base_with_regimes()
    rc.regimes.vix.resume_consec = 0
    with pytest.raises(ConfigError, match="vix.resume_consec"):
        validate_run_config(rc)


def test_validation_rule_14_circuit_breaker_trip_pct_negative():
    rc = _base_with_regimes()
    rc.regimes.circuit_breaker.trip_pct = 0.05
    with pytest.raises(ConfigError, match="circuit_breaker.trip_pct"):
        validate_run_config(rc)


# Task 14: Universe-membership validation rules (rules 15-18)
def test_validation_rule_15_universe_path_exists(tmp_path):
    rc = _base_run_config()
    rc.universe_path = str(tmp_path / "missing.yaml")
    rc.data.symbols = []  # clear, otherwise rule 18 fires first
    with pytest.raises(ConfigError, match="universe_path"):
        validate_run_config(rc)


def test_validation_rule_16_overrides_subset_of_strategy_params(tmp_path):
    universe_yaml = tmp_path / "universe.yaml"
    universe_yaml.write_text(
        "universe:\n"
        "  SPY: {sector: Index, overrides: {nonexistent_key: 1}}\n"
    )
    rc = _base_run_config()
    rc.universe_path = str(universe_yaml)
    rc.data.symbols = []
    with pytest.raises(ConfigError, match="overrides"):
        validate_run_config(rc)


def test_validation_rule_17_aux_symbols_required_when_regimes_enabled():
    rc = _base_with_regimes()
    rc.regimes.spy_ema.enabled = True
    rc.data.aux_symbols = []
    with pytest.raises(ConfigError, match="aux_symbols.*SPY"):
        validate_run_config(rc)


def test_validation_rule_18_symbols_and_universe_path_mutex(tmp_path):
    universe_yaml = tmp_path / "universe.yaml"
    universe_yaml.write_text("universe:\n  SPY: {sector: Index}\n")
    rc = _base_run_config()
    rc.universe_path = str(universe_yaml)
    # data.symbols already has SPY from _base_run_config
    with pytest.raises(ConfigError, match="mutually exclusive"):
        validate_run_config(rc)
