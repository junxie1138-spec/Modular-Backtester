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

    if rc.portfolio.size <= 0 or rc.portfolio.size > 1.0:
        raise ConfigError("portfolio.size must be in (0, 1] when sizing_mode is percent_equity")

    if rc.wfo and rc.wfo.enabled:
        for k in ("train_bars", "test_bars", "step_bars"):
            v = getattr(rc.wfo, k)
            if v is None or v <= 0:
                raise ConfigError(f"wfo.{k} must be a positive integer when wfo.enabled")
