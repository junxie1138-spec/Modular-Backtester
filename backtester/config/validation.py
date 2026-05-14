from __future__ import annotations

from pathlib import Path

import pandas as pd
import yaml

from backtester.config.models import RunConfig
from backtester.core.exceptions import ConfigError


def validate_run_config(rc: RunConfig) -> None:
    if not rc.run_name:
        raise ConfigError("run_name must be non-empty")
    if not rc.strategy:
        raise ConfigError("strategy must be non-empty")

    # Allow empty symbols if universe_path is provided (v0.4.0)
    if not rc.data.symbols and not rc.universe_path:
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
    # v0.4.0 regime gates validation
    _validate_regimes(rc)
    # v0.4.0 universe membership validation
    _validate_universe_path(rc)
    _validate_aux_symbols(rc)


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


def _validate_regimes(rc: RunConfig) -> None:
    """Validate regime gates (SPY pcts, VIX hysteresis, circuit breaker)."""
    if rc.regimes is None:
        return
    r = rc.regimes
    if r.circuit_breaker.pause_days < 0:
        raise ConfigError(f"regimes.circuit_breaker.pause_days must be >= 0; got {r.circuit_breaker.pause_days}")
    if r.vix.resume_threshold >= r.vix.trip_threshold:
        raise ConfigError(
            f"regimes.vix.resume_threshold ({r.vix.resume_threshold}) must be < "
            f"trip_threshold ({r.vix.trip_threshold})"
        )
    if r.spy_ema.trip_pct > 0:
        raise ConfigError(f"regimes.spy_ema.trip_pct must be <= 0; got {r.spy_ema.trip_pct}")
    if r.spy_ema.resume_pct < 0:
        raise ConfigError(f"regimes.spy_ema.resume_pct must be >= 0; got {r.spy_ema.resume_pct}")
    if r.vix.trip_consec < 1:
        raise ConfigError(f"regimes.vix.trip_consec must be >= 1; got {r.vix.trip_consec}")
    if r.vix.resume_consec < 1:
        raise ConfigError(f"regimes.vix.resume_consec must be >= 1; got {r.vix.resume_consec}")
    if r.circuit_breaker.trip_pct >= 0:
        raise ConfigError(f"regimes.circuit_breaker.trip_pct must be < 0; got {r.circuit_breaker.trip_pct}")


def _validate_universe_path(rc: RunConfig) -> None:
    """Validate universe-membership rules (path exists, overrides subset)."""
    if rc.universe_path is None:
        return
    if rc.data.symbols:
        raise ConfigError(
            "data.symbols and universe_path are mutually exclusive; universe.yaml "
            "is the single source of symbol membership for multi-symbol runs"
        )
    if not Path(rc.universe_path).exists():
        raise ConfigError(f"universe_path does not exist: {rc.universe_path}")
    with open(rc.universe_path, encoding="utf-8") as f:
        doc = yaml.safe_load(f)
    universe = (doc or {}).get("universe", {})
    allowed_keys = set(rc.strategy_params.keys())
    for sym, meta in universe.items():
        overrides = (meta or {}).get("overrides", {}) or {}
        unknown = set(overrides) - allowed_keys
        if unknown:
            raise ConfigError(
                f"universe.yaml: {sym} overrides reference keys not in strategy_params: {sorted(unknown)}"
            )


def _validate_aux_symbols(rc: RunConfig) -> None:
    """Validate aux_symbols required by enabled regimes."""
    if rc.regimes is None:
        return
    required = []
    if rc.regimes.spy_ema.enabled:
        required.append("SPY")
    if rc.regimes.vix.enabled:
        required.append("^VIX")
    missing = [s for s in required if s not in rc.data.aux_symbols]
    if missing:
        raise ConfigError(
            f"data.aux_symbols must include {missing} because regimes are enabled "
            f"that depend on them"
        )
