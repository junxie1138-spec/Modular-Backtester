from __future__ import annotations

import pandas as pd

from backtester.config.models import RunConfig
from backtester.core.exceptions import ConfigError


def validate_run_config(rc: RunConfig) -> None:
    if not rc.run_name:
        raise ConfigError("run_name must be non-empty")
    if not rc.strategy:
        raise ConfigError("strategy must be non-empty")

    if not rc.data.symbols:
        raise ConfigError("data.symbols must contain at least one symbol")
    try:
        start = pd.Timestamp(rc.data.start)
        end = pd.Timestamp(rc.data.end)
    except Exception as exc:
        raise ConfigError(f"invalid data.start / data.end: {exc}") from exc
    if start >= end:
        raise ConfigError("data.start must be strictly before data.end")

    if rc.execution.initial_cash <= 0:
        raise ConfigError("execution.initial_cash must be > 0")
    if rc.execution.commission_bps < 0 or rc.execution.slippage_bps < 0:
        raise ConfigError("execution commission_bps and slippage_bps must be >= 0")

    # Trailing-stop validation.
    pct = rc.execution.trailing_stop_pct
    atr_mult = rc.execution.trailing_stop_atr_mult
    atr_period = rc.execution.trailing_stop_atr_period
    if pct is not None and atr_mult is not None:
        raise ConfigError(
            "execution.trailing_stop_pct and trailing_stop_atr_mult are mutually exclusive"
        )
    if pct is not None and not (0.0 < pct < 1.0):
        raise ConfigError("execution.trailing_stop_pct must be in (0, 1)")
    if atr_mult is not None and atr_mult <= 0.0:
        raise ConfigError("execution.trailing_stop_atr_mult must be > 0")
    if atr_period < 2:
        raise ConfigError("execution.trailing_stop_atr_period must be >= 2")

    if rc.portfolio.size <= 0 or rc.portfolio.size > 1.0:
        raise ConfigError("portfolio.size must be in (0, 1] when sizing_mode is percent_equity")

    if rc.wfo and rc.wfo.enabled:
        for k in ("train_bars", "test_bars", "step_bars"):
            v = getattr(rc.wfo, k)
            if v is None or v <= 0:
                raise ConfigError(f"wfo.{k} must be a positive integer when wfo.enabled")

    # v0.4.0 tranche-stop validation
    _validate_tranche_stop(rc)
    # v0.4.0 portfolio sizing validation
    _validate_portfolio_sizing(rc)


def _validate_tranche_stop(rc: RunConfig) -> None:
    """Validate tranche-stop (hard/runner) mutual exclusion and bounds."""
    ex = rc.execution
    has_v030 = ex.trailing_stop_pct is not None or ex.trailing_stop_atr_mult is not None
    has_hard = ex.hard_stop_atr_mult is not None
    has_runner = ex.runner_atr_mult is not None

    if has_hard != has_runner:
        raise ConfigError(
            "execution: hard_stop_atr_mult and runner_atr_mult are both-or-neither"
        )
    if has_hard and has_v030:
        raise ConfigError(
            "execution: v0.3.0 trailing_stop_* keys and v0.4.0 hard/runner keys are mutually exclusive"
        )
    if has_hard:
        if ex.hard_stop_atr_mult <= 0:
            raise ConfigError("execution.hard_stop_atr_mult must be > 0")
        if ex.runner_atr_mult <= 0:
            raise ConfigError("execution.runner_atr_mult must be > 0")
        if ex.tranche_stop_atr_period < 2:
            raise ConfigError("execution.tranche_stop_atr_period must be >= 2")


def _validate_portfolio_sizing(rc: RunConfig) -> None:
    """Validate portfolio sizing bounds (caps, reserve, budget, sector)."""
    p = rc.portfolio
    if not (0 < p.position_cap_pct <= 1):
        raise ConfigError(f"portfolio.position_cap_pct must be in (0, 1]; got {p.position_cap_pct}")
    if not (0 <= p.cash_reserve_pct < 1):
        raise ConfigError(f"portfolio.cash_reserve_pct must be in [0, 1); got {p.cash_reserve_pct}")
    if not (0 < p.risk_budget_pct <= 1):
        raise ConfigError(f"portfolio.risk_budget_pct must be in (0, 1]; got {p.risk_budget_pct}")
    if not (0 < p.sector_cap_pct <= 1):
        raise ConfigError(f"portfolio.sector_cap_pct must be in (0, 1]; got {p.sector_cap_pct}")
    if p.sizing_mode == "vol_targeted" and p.vol_target <= 0:
        raise ConfigError(f"portfolio.vol_target must be > 0 when sizing_mode='vol_targeted'")
