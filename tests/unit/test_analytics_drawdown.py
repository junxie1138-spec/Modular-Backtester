from __future__ import annotations

import pandas as pd
import pytest

from backtester.analytics.drawdown import drawdown_series, max_drawdown


def test_no_drawdown_when_monotonic():
    eq = pd.Series([1.0, 1.1, 1.2, 1.3])
    dd = drawdown_series(eq)
    assert (dd == 0).all()
    assert max_drawdown(eq) == 0.0


def test_max_drawdown_basic():
    eq = pd.Series([1.0, 1.2, 0.9, 1.1, 0.6, 1.0])
    # peak at 1.2, trough at 0.6 -> -50%
    assert max_drawdown(eq) == pytest.approx(-0.5)


def test_drawdown_series_lengths_match():
    eq = pd.Series([1.0, 1.1, 0.9, 1.05, 0.8])
    assert len(drawdown_series(eq)) == len(eq)
