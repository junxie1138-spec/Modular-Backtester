from __future__ import annotations

import pandas as pd


def extract_round_trips(trades: pd.DataFrame) -> pd.DataFrame:
    """Pair sequential BUY/SELL trades into long-only round trips."""
    if len(trades) == 0:
        return pd.DataFrame(columns=[
            "entry_time", "exit_time", "qty", "entry_price", "exit_price",
            "pnl", "return_pct", "bars_held",
        ])

    rows = []
    open_buy: dict | None = None
    for _, t in trades.iterrows():
        if t["side"] == "buy":
            open_buy = {"time": t["timestamp"], "qty": t["qty"], "price": t["price"],
                        "commission": t["commission"]}
        elif t["side"] == "sell" and open_buy is not None:
            pnl = (t["price"] - open_buy["price"]) * t["qty"] - (t["commission"] + open_buy["commission"])
            ret = (t["price"] / open_buy["price"]) - 1.0
            bars_held = max(1, (t["timestamp"] - open_buy["time"]).days)
            rows.append({
                "entry_time": open_buy["time"],
                "exit_time": t["timestamp"],
                "qty": t["qty"],
                "entry_price": open_buy["price"],
                "exit_price": t["price"],
                "pnl": pnl,
                "return_pct": ret,
                "bars_held": bars_held,
            })
            open_buy = None

    return pd.DataFrame(rows)
