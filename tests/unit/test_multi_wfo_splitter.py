from __future__ import annotations

import pandas as pd
import pytest


def _ohlcv(closes, start="2020-01-02"):
    idx = pd.date_range(start, periods=len(closes), freq="B")
    return pd.DataFrame(
        {"open": closes, "high": [c + 0.5 for c in closes], "low": [c - 0.5 for c in closes],
         "close": closes, "volume": [1_000_000] * len(closes)},
        index=idx,
    )


def test_splitter_yields_expected_windows():
    from backtester.wfo.multi_splitter import MultiSymbolWFOSplitter, WindowPanel
    splitter = MultiSymbolWFOSplitter(train_bars=100, test_bars=50, step_bars=50)
    data = {"AAA": _ohlcv([100.0] * 300), "BBB": _ohlcv([200.0] * 300)}
    aux_data = {"SPY": _ohlcv([400.0] * 300)}
    windows = list(splitter.split(data=data, aux_data=aux_data))
    # Windows: (0-99 train, 100-149 test), (50-149, 150-199), (100-199, 200-249),
    #          (150-249, 250-299). 4 windows total.
    assert len(windows) == 4
    assert windows[0].train_data["AAA"].iloc[0].name == data["AAA"].index[0]
    assert len(windows[0].train_data["AAA"]) == 100
    assert len(windows[0].test_data["AAA"]) == 50


def test_splitter_slices_all_symbols_together():
    from backtester.wfo.multi_splitter import MultiSymbolWFOSplitter
    splitter = MultiSymbolWFOSplitter(train_bars=50, test_bars=20, step_bars=20)
    data = {f"S{i}": _ohlcv([100.0 + i] * 200) for i in range(5)}
    aux_data = {"SPY": _ohlcv([400.0] * 200)}
    windows = list(splitter.split(data=data, aux_data=aux_data))
    for w in windows:
        # All symbols and aux share the same train_start and train_end.
        train_starts = {sym: df.index[0] for sym, df in w.train_data.items()}
        aux_starts = {sym: df.index[0] for sym, df in w.train_aux.items()}
        assert len(set(train_starts.values())) == 1
        assert list(aux_starts.values())[0] == list(train_starts.values())[0]


def test_splitter_stops_when_insufficient_data():
    from backtester.wfo.multi_splitter import MultiSymbolWFOSplitter
    splitter = MultiSymbolWFOSplitter(train_bars=80, test_bars=20, step_bars=20)
    data = {"AAA": _ohlcv([100.0] * 100)}
    aux_data = {}
    windows = list(splitter.split(data=data, aux_data=aux_data))
    # 80 + 20 = 100 bars exactly fit one window.
    assert len(windows) == 1
