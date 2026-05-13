from __future__ import annotations

import pandas as pd

from backtester.analytics.trades import extract_round_trips


def _trades_df(rows):
    return pd.DataFrame(rows, columns=["timestamp", "side", "qty", "price", "commission", "notional"])


def test_no_trades_returns_empty():
    rt = extract_round_trips(_trades_df([]))
    assert len(rt) == 0


def test_single_round_trip():
    rows = [
        {"timestamp": pd.Timestamp("2024-01-02"), "side": "buy",  "qty": 10, "price": 100.0, "commission": 0.0, "notional": 1000.0},
        {"timestamp": pd.Timestamp("2024-01-10"), "side": "sell", "qty": 10, "price": 110.0, "commission": 0.0, "notional": 1100.0},
    ]
    rt = extract_round_trips(_trades_df(rows))
    assert len(rt) == 1
    row = rt.iloc[0]
    assert row["pnl"] == 100.0
    assert row["return_pct"] > 0
    assert row["bars_held"] >= 1


def test_two_round_trips_sequential():
    rows = [
        {"timestamp": pd.Timestamp("2024-01-02"), "side": "buy",  "qty": 10, "price": 100.0, "commission": 0.0, "notional": 1000.0},
        {"timestamp": pd.Timestamp("2024-01-10"), "side": "sell", "qty": 10, "price": 90.0,  "commission": 0.0, "notional": 900.0},
        {"timestamp": pd.Timestamp("2024-02-01"), "side": "buy",  "qty": 5,  "price": 50.0,  "commission": 0.0, "notional": 250.0},
        {"timestamp": pd.Timestamp("2024-02-10"), "side": "sell", "qty": 5,  "price": 60.0,  "commission": 0.0, "notional": 300.0},
    ]
    rt = extract_round_trips(_trades_df(rows))
    assert len(rt) == 2
    assert rt.iloc[0]["pnl"] == -100.0
    assert rt.iloc[1]["pnl"] == 50.0
