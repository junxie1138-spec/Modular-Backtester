from __future__ import annotations

import pandas as pd
import pytest

from backtester.core.types import StrategyContext, SignalFrame, BacktestResult


def test_strategy_context_defaults():
    ctx = StrategyContext(symbol="SPY", timeframe="1d", warmup_bars=20)
    assert ctx.symbol == "SPY"
    assert ctx.metadata == {}


def test_strategy_context_metadata_is_per_instance():
    a = StrategyContext(symbol="A", timeframe="1d", warmup_bars=1)
    b = StrategyContext(symbol="B", timeframe="1d", warmup_bars=1)
    a.metadata["k"] = "v"
    assert b.metadata == {}


def test_signal_frame_defaults():
    df = pd.DataFrame({"signal": [0, 1], "size": [1.0, 1.0]})
    sf = SignalFrame(data=df)
    assert sf.signal_column == "signal"
    assert sf.size_column == "size"
    assert sf.price_column is None


def test_backtest_result_holds_frames():
    summary = {"total_return": 0.1}
    eq = pd.DataFrame({"equity": [1.0]})
    trades = pd.DataFrame({"pnl": [0.0]})
    positions = pd.DataFrame({"qty": [0]})
    r = BacktestResult(summary=summary, equity_curve=eq, trades=trades, positions=positions)
    assert r.summary["total_return"] == 0.1
    assert len(r.equity_curve) == 1
