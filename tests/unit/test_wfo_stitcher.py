from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from backtester.core.types import BacktestResult
from backtester.wfo.stitcher import WalkForwardStitcher


def _result(start: str, n: int, equity_start: float, equity_end: float) -> BacktestResult:
    idx = pd.bdate_range(start, periods=n)
    eq = pd.DataFrame({"equity": pd.Series([equity_start, *([0] * (n - 2)), equity_end]).interpolate().values},
                      index=idx)
    trades = pd.DataFrame([
        {"timestamp": idx[0], "side": "buy", "qty": 1, "price": 100.0, "commission": 0, "notional": 100},
        {"timestamp": idx[-1], "side": "sell", "qty": 1, "price": 110.0, "commission": 0, "notional": 110},
    ])
    positions = pd.DataFrame({"qty": [1] * n}, index=idx)
    return BacktestResult(
        summary={"total_return": equity_end / equity_start - 1.0, "sharpe": 1.0,
                 "max_drawdown": -0.05, "n_trades": 2},
        equity_curve=eq, trades=trades, positions=positions,
    )


def test_stitcher_combines_oos_equity():
    windows = [
        {"train_start": pd.Timestamp("2024-01-01"), "train_end": pd.Timestamp("2024-03-01"),
         "test_start": pd.Timestamp("2024-03-04"), "test_end": pd.Timestamp("2024-04-01"),
         "best_params": {"fast": 10}, "train_summary": {"sharpe": 1.5},
         "test_summary": {"total_return": 0.05, "sharpe": 1.0, "max_drawdown": -0.02, "n_trades": 2, "n_round_trips": 1, "win_rate": 1.0, "avg_round_trip_pnl": 100, "annualized_return": 0.6, "annualized_vol": 0.2, "sortino": 1.1, "time_in_market": 1.0, "turnover": 1.0, "final_equity": 10500},
         "test_result": _result("2024-03-04", 20, 10_000, 10_500)},
        {"train_start": pd.Timestamp("2024-02-01"), "train_end": pd.Timestamp("2024-04-01"),
         "test_start": pd.Timestamp("2024-04-02"), "test_end": pd.Timestamp("2024-05-01"),
         "best_params": {"fast": 20}, "train_summary": {"sharpe": 1.6},
         "test_summary": {"total_return": -0.02, "sharpe": -0.2, "max_drawdown": -0.05, "n_trades": 2, "n_round_trips": 1, "win_rate": 0.0, "avg_round_trip_pnl": -200, "annualized_return": -0.3, "annualized_vol": 0.2, "sortino": -0.3, "time_in_market": 1.0, "turnover": 1.0, "final_equity": 9800},
         "test_result": _result("2024-04-02", 20, 10_500, 10_290)},
    ]
    out = WalkForwardStitcher().combine(windows)
    assert "oos_equity_curve" in out
    assert "oos_summary" in out
    assert "is_summary_avg" in out
    assert "parameter_stability" in out
    assert len(out["oos_equity_curve"]) == 40  # 20 + 20
    assert out["oos_equity_curve"]["equity"].iloc[-1] > 0
    assert out["parameter_stability"]["fast"]["unique"] == 2


def test_stitcher_handles_single_window():
    windows = [{
        "train_start": pd.Timestamp("2024-01-01"), "train_end": pd.Timestamp("2024-03-01"),
        "test_start": pd.Timestamp("2024-03-04"), "test_end": pd.Timestamp("2024-04-01"),
        "best_params": {"fast": 10}, "train_summary": {"sharpe": 1.5},
        "test_summary": {"total_return": 0.05, "sharpe": 1.0, "max_drawdown": -0.02, "n_trades": 2, "n_round_trips": 1, "win_rate": 1.0, "avg_round_trip_pnl": 100, "annualized_return": 0.6, "annualized_vol": 0.2, "sortino": 1.1, "time_in_market": 1.0, "turnover": 1.0, "final_equity": 10500},
        "test_result": _result("2024-03-04", 20, 10_000, 10_500),
    }]
    out = WalkForwardStitcher().combine(windows)
    assert len(out["oos_equity_curve"]) == 20


def test_stitcher_skips_non_numeric_is_summary_keys_without_warning():
    """Real train_summary dicts (from BacktestEngine) include non-numeric keys
    like 'params' (dict), 'symbol' (str), 'timeframe' (str) alongside numeric
    metrics. The stitcher must skip those when averaging in-sample metrics
    rather than computing np.mean([]) and emitting 'Mean of empty slice'."""
    import warnings as _warnings

    realistic_train_summary = {
        "sharpe": 1.5,
        "total_return": 0.10,
        "n_trades": 4,
        "params": {"fast": 10, "slow": 30},  # non-numeric
        "symbol": "SYN",                       # non-numeric
        "timeframe": "1d",                     # non-numeric
    }
    windows = [{
        "train_start": pd.Timestamp("2024-01-01"), "train_end": pd.Timestamp("2024-03-01"),
        "test_start": pd.Timestamp("2024-03-04"), "test_end": pd.Timestamp("2024-04-01"),
        "best_params": {"fast": 10},
        "train_summary": realistic_train_summary,
        "test_summary": {"sharpe": 1.0},
        "test_result": _result("2024-03-04", 20, 10_000, 10_500),
    }]

    with _warnings.catch_warnings():
        # Only the "Mean of empty slice" warning is in scope here; the
        # pandas nanstd path emits unrelated RuntimeWarnings on tiny equity
        # series that aren't being fixed by this change.
        _warnings.filterwarnings("error", message="Mean of empty slice")
        out = WalkForwardStitcher().combine(windows)

    is_avg = out["is_summary_avg"]
    # Numeric keys averaged through; non-numeric ones skipped entirely.
    assert is_avg["sharpe"] == 1.5
    assert is_avg["total_return"] == 0.10
    assert is_avg["n_trades"] == 4
    assert "params" not in is_avg
    assert "symbol" not in is_avg
    assert "timeframe" not in is_avg


def test_combine_threads_timeframe_into_oos_summary() -> None:
    from types import SimpleNamespace
    from backtester.wfo.stitcher import WalkForwardStitcher
    rng = np.random.default_rng(2)
    vals = 100 * np.exp(np.cumsum(rng.normal(0.0003, 0.01, 400)))
    idx = pd.bdate_range("2024-01-02", periods=400)
    eq = pd.DataFrame({"equity": vals}, index=idx)
    test_result = SimpleNamespace(
        equity_curve=eq, trades=pd.DataFrame(),
        positions=pd.DataFrame({"qty": [0] * 400}, index=idx),
    )
    window = {
        "window_index": 0, "test_result": test_result,
        "train_summary": {"sharpe": 1.0}, "best_params": {"fast": 10},
    }
    st = WalkForwardStitcher()
    sh_d = st.combine([window], timeframe="1d")["oos_summary"]["sharpe"]
    sh_h = st.combine([window], timeframe="1h")["oos_summary"]["sharpe"]
    assert sh_h == pytest.approx(sh_d * math.sqrt(1638 / 252), rel=1e-6)
