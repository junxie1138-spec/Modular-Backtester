from __future__ import annotations

import pandas as pd
import pytest

from backtester.analytics.exposure import time_in_market, turnover


def test_time_in_market_zero_when_always_flat():
    positions = pd.DataFrame({"qty": [0, 0, 0, 0]})
    assert time_in_market(positions) == 0.0


def test_time_in_market_full_when_always_long():
    positions = pd.DataFrame({"qty": [1, 1, 1, 1]})
    assert time_in_market(positions) == 1.0


def test_time_in_market_half():
    positions = pd.DataFrame({"qty": [0, 1, 0, 1]})
    assert time_in_market(positions) == pytest.approx(0.5)


def test_turnover_zero_with_no_trades():
    trades = pd.DataFrame(columns=["notional"])
    eq = pd.DataFrame({"equity": [10_000.0, 10_000.0]})
    assert turnover(trades, eq) == 0.0


def test_turnover_basic():
    trades = pd.DataFrame({"notional": [10_000.0, 10_000.0]})
    eq = pd.DataFrame({"equity": [10_000.0, 10_500.0]})
    # turnover = sum(notional) / mean(equity)
    assert turnover(trades, eq) == pytest.approx(20_000.0 / 10_250.0)
