from __future__ import annotations

import pytest

from backtester.wfo.splitter import Window, WalkForwardSplitter
from tests.fixtures.synthetic import make_ohlcv


def test_window_dataclass_fields():
    data = make_ohlcv(n=10, seed=0)
    w = Window(train_data=data.iloc[:5], test_data=data.iloc[5:])
    assert len(w.train_data) == 5
    assert len(w.test_data) == 5


def test_splitter_produces_expected_windows():
    data = make_ohlcv(n=1000, seed=0)
    splitter = WalkForwardSplitter()
    windows = splitter.split(data=data, train_bars=252, test_bars=63, step_bars=63)
    assert len(windows) >= 10
    for w in windows:
        assert len(w.train_data) == 252
        assert len(w.test_data) <= 63


def test_splitter_train_precedes_test():
    data = make_ohlcv(n=500, seed=0)
    splitter = WalkForwardSplitter()
    windows = splitter.split(data=data, train_bars=200, test_bars=50, step_bars=50)
    for w in windows:
        assert w.train_data.index.max() < w.test_data.index.min()


def test_splitter_steps_advance():
    data = make_ohlcv(n=600, seed=0)
    splitter = WalkForwardSplitter()
    windows = splitter.split(data=data, train_bars=200, test_bars=50, step_bars=50)
    starts = [w.train_data.index.min() for w in windows]
    assert starts == sorted(set(starts)) and len(starts) == len(windows)


def test_splitter_raises_on_too_small_data():
    data = make_ohlcv(n=100, seed=0)
    with pytest.raises(ValueError, match="too short"):
        WalkForwardSplitter().split(data=data, train_bars=200, test_bars=50, step_bars=50)
