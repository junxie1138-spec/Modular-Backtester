from __future__ import annotations

import pandas as pd


def time_in_market(positions: pd.DataFrame) -> float:
    if len(positions) == 0:
        return 0.0
    return float((positions["qty"].abs() > 0).mean())


def turnover(trades: pd.DataFrame, equity_curve: pd.DataFrame) -> float:
    if len(trades) == 0:
        return 0.0
    avg_equity = float(equity_curve["equity"].mean())
    if avg_equity == 0:
        return 0.0
    return float(trades["notional"].abs().sum() / avg_equity)
