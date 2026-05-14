from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass(slots=True)
class DataConfig:
    symbols: List[str]
    timeframe: str
    start: str
    end: str
    source: str = "csv"
    root: str = "data/raw"
    # v0.4.0 additions:
    auto_adjust: bool = True
    aux_symbols: List[str] = field(default_factory=list)


@dataclass(slots=True)
class ExecutionConfig:
    initial_cash: float = 100_000.0
    commission_bps: float = 1.0
    slippage_bps: float = 2.0
    allow_fractional: bool = False
    allow_short: bool = False
    trailing_stop_pct: Optional[float] = None
    trailing_stop_atr_mult: Optional[float] = None
    trailing_stop_atr_period: int = 14
    # v0.4.0 additions to ExecutionConfig:
    hard_stop_atr_mult: Optional[float] = None
    runner_atr_mult: Optional[float] = None
    breakeven_floor: bool = True
    tranche_stop_atr_period: int = 20


@dataclass(slots=True)
class PortfolioConfig:
    sizing_mode: str = "percent_equity"
    size: float = 1.0
    # v0.4.0 additions:
    vol_target: float = 0.12
    position_cap_pct: float = 1.0
    cash_reserve_pct: float = 0.0
    risk_budget_pct: float = 1.0
    sector_cap_pct: float = 1.0


@dataclass(slots=True)
class SpyEmaRegimeConfig:
    enabled: bool = False
    ema_lookback: int = 200
    trip_pct: float = -0.02
    resume_pct: float = 0.02


@dataclass(slots=True)
class VixRegimeConfig:
    enabled: bool = False
    trip_threshold: float = 30.0
    trip_consec: int = 2
    resume_threshold: float = 25.0
    resume_consec: int = 3


@dataclass(slots=True)
class CircuitBreakerConfig:
    enabled: bool = False
    pnl_window_days: int = 20
    trip_pct: float = -0.05
    pause_days: int = 10


@dataclass(slots=True)
class RegimesConfig:
    spy_ema: SpyEmaRegimeConfig = field(default_factory=SpyEmaRegimeConfig)
    vix: VixRegimeConfig = field(default_factory=VixRegimeConfig)
    circuit_breaker: CircuitBreakerConfig = field(default_factory=CircuitBreakerConfig)


@dataclass(slots=True)
class OptimizationConfig:
    objective: str = "sharpe"
    param_space: Dict[str, List[Any]] = field(default_factory=dict)


@dataclass(slots=True)
class WFOConfig:
    enabled: bool = False
    train_bars: Optional[int] = None
    test_bars: Optional[int] = None
    step_bars: Optional[int] = None


@dataclass(slots=True)
class RunConfig:
    run_name: str
    strategy: str
    strategy_params: Dict[str, Any]
    data: DataConfig
    execution: ExecutionConfig
    portfolio: PortfolioConfig
    optimization: Optional[OptimizationConfig] = None
    wfo: Optional[WFOConfig] = None
    output_root: str = "output/runs"
    seed: int = 0
    # v0.4.0 additions to RunConfig:
    universe_path: Optional[str] = None
    regimes: Optional[RegimesConfig] = None
