from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional
import pandas as pd


@dataclass(slots=True)
class StrategyContext:
    symbol: str
    timeframe: str
    warmup_bars: int
    metadata: Dict[str, Any] = field(default_factory=dict)

    # v0.4.0 additions: simulator-populated state visible to strategies.
    position_phase: Dict[str, Any] = field(default_factory=dict)
    bars_in_phase: Dict[str, int] = field(default_factory=dict)
    recent_pnl: Optional[pd.Series] = None
    regime: Optional[Any] = None  # RegimeState — typed in Phase 8


@dataclass(slots=True)
class SignalFrame:
    data: pd.DataFrame
    signal_column: str = "signal"
    size_column: Optional[str] = "size"
    price_column: Optional[str] = None


@dataclass(slots=True)
class BacktestResult:
    summary: Dict[str, Any]
    equity_curve: pd.DataFrame
    trades: pd.DataFrame
    positions: pd.DataFrame
