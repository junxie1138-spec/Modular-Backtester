from __future__ import annotations

import pandas as pd


def compute_summary_metrics(equity_curve: pd.DataFrame, trades: pd.DataFrame, positions: pd.DataFrame) -> dict:
    if len(equity_curve) == 0:
        return {"total_return": 0.0, "n_trades": 0}
    start = equity_curve["equity"].iloc[0]
    end = equity_curve["equity"].iloc[-1]
    return {
        "total_return": (end / start) - 1.0,
        "n_trades": int(len(trades)),
        "final_equity": float(end),
    }
