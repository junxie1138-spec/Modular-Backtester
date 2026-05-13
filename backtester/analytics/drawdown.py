from __future__ import annotations

import pandas as pd


def drawdown_series(equity: pd.Series) -> pd.Series:
    running_max = equity.cummax()
    return (equity / running_max) - 1.0


def max_drawdown(equity: pd.Series) -> float:
    if len(equity) == 0:
        return 0.0
    return float(drawdown_series(equity).min())
