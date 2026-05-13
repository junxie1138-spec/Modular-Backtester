from __future__ import annotations

from tests.fixtures.synthetic import make_ohlcv


def test_make_ohlcv_shape_and_columns():
    df = make_ohlcv(n=100, seed=42)
    assert len(df) == 100
    assert list(df.columns) == ["open", "high", "low", "close", "volume"]


def test_make_ohlcv_is_deterministic():
    a = make_ohlcv(n=50, seed=7)
    b = make_ohlcv(n=50, seed=7)
    assert (a.values == b.values).all()


def test_make_ohlcv_index_is_business_days():
    df = make_ohlcv(n=20, seed=0, start="2024-01-02")
    assert df.index.is_monotonic_increasing
    assert df.index.inferred_freq == "B"


def test_make_ohlcv_high_low_invariants():
    df = make_ohlcv(n=200, seed=1)
    assert (df["high"] >= df["low"]).all()
    assert (df["high"] >= df["open"]).all()
    assert (df["high"] >= df["close"]).all()
    assert (df["low"] <= df["open"]).all()
    assert (df["low"] <= df["close"]).all()
    assert (df["volume"] > 0).all()
