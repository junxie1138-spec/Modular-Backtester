from __future__ import annotations

import pandas as pd
import pytest


def _make_window_result(idx, start, end, eq_values):
    from backtester.wfo.multi_runner import WindowResult
    idx_dates = pd.date_range(start, periods=len(eq_values), freq="B")
    return WindowResult(
        window_idx=idx,
        train_start=pd.Timestamp(start) - pd.Timedelta(days=30),
        train_end=pd.Timestamp(start) - pd.Timedelta(days=1),
        test_start=pd.Timestamp(start),
        test_end=pd.Timestamp(end),
        best_params={"threshold": 1.0},
        is_summary={"portfolio_total_return": 0.05},
        oos_summary={"portfolio_total_return": 0.02},
        oos_equity_curve=pd.Series(eq_values, index=idx_dates),
    )


def test_stitcher_scales_consecutive_windows():
    from backtester.wfo.multi_stitcher import MultiSymbolWFOStitcher
    w1 = _make_window_result(0, "2024-01-02", "2024-01-10", [100, 101, 102, 103, 104, 105])
    w2 = _make_window_result(1, "2024-02-01", "2024-02-08", [100, 99, 98, 99, 100])
    stitcher = MultiSymbolWFOStitcher()
    result = stitcher.stitch([w1, w2])
    # Stitched curve has 11 bars. First piece (6) ends at 105. Second piece (5) starts at 100,
    # gets scaled by 105/100=1.05, so 100*1.05=105, 99*1.05, 98*1.05, 99*1.05, 100*1.05.
    assert len(result.oos_equity_curve) == 11
    assert result.oos_equity_curve.iloc[5] == pytest.approx(105.0)
    assert result.oos_equity_curve.iloc[6] == pytest.approx(105.0)
    assert result.oos_equity_curve.iloc[-1] == pytest.approx(105.0)


def test_stitcher_emits_parameter_stability():
    from backtester.wfo.multi_stitcher import MultiSymbolWFOStitcher
    w1 = _make_window_result(0, "2024-01-02", "2024-01-10", [100.0] * 5)
    w1.best_params = {"threshold": 1.0, "mean_lookback": 10}
    w2 = _make_window_result(1, "2024-02-01", "2024-02-08", [100.0] * 5)
    w2.best_params = {"threshold": 1.5, "mean_lookback": 14}
    stitcher = MultiSymbolWFOStitcher()
    result = stitcher.stitch([w1, w2])
    assert result.parameter_stability["threshold"] == [1.0, 1.5]
    assert result.parameter_stability["mean_lookback"] == [10, 14]
