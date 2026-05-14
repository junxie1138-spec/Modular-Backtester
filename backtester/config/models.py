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


@dataclass(slots=True)
class PortfolioConfig:
    sizing_mode: str = "percent_equity"
    size: float = 1.0


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
