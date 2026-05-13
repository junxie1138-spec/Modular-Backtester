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
