from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from backtester.analytics.metrics import (
    compute_summary_metrics,
    annualized_return,
    annualized_volatility,
    sharpe_ratio,
    sortino_ratio,
)


def _eq(values):
    idx = pd.bdate_range("2024-01-02", periods=len(values))
    return pd.DataFrame({"equity": values, "cash": values, "position_value": [0] * len(values)}, index=idx)


def test_total_return_basic():
    eq = _eq([100.0, 110.0, 120.0])
    out = compute_summary_metrics(eq, pd.DataFrame(), pd.DataFrame({"qty": [0, 1, 1]}))
    assert out["total_return"] == pytest.approx(0.2)


def test_annualized_return_one_year():
    n = 252
    eq = pd.Series(np.linspace(100, 110, n))
    ar = annualized_return(eq)
    assert ar == pytest.approx(0.10, rel=1e-2)


def test_volatility_nonzero_when_returns_vary():
    rng = np.random.default_rng(0)
    eq = pd.Series(100 * np.exp(np.cumsum(rng.normal(0, 0.01, 500))))
    assert annualized_volatility(eq) > 0


def test_sharpe_zero_when_no_variance():
    eq = pd.Series([100.0] * 100)
    assert sharpe_ratio(eq) == 0.0


def test_sortino_finite():
    rng = np.random.default_rng(0)
    eq = pd.Series(100 * np.exp(np.cumsum(rng.normal(0.0005, 0.01, 500))))
    s = sortino_ratio(eq)
    assert math.isfinite(s)


def test_summary_keys_present():
    eq = _eq(np.linspace(100, 130, 252).tolist())
    trades = pd.DataFrame({
        "timestamp": [eq.index[10], eq.index[20]],
        "side": ["buy", "sell"],
        "qty": [10, 10],
        "price": [100.0, 110.0],
        "commission": [0.0, 0.0],
        "notional": [1000.0, 1100.0],
    })
    positions = pd.DataFrame({"qty": [0] * 252}, index=eq.index)
    positions.iloc[10:20, 0] = 10
    out = compute_summary_metrics(eq, trades, positions)
    for key in ["total_return", "annualized_return", "annualized_vol",
                "sharpe", "sortino", "max_drawdown", "n_trades",
                "n_round_trips", "win_rate", "avg_round_trip_pnl",
                "time_in_market", "turnover", "final_equity"]:
        assert key in out, f"missing {key}"
