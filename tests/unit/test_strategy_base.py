from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
import pytest

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class _DummyParams:
    lookback: int = 5


class _DummyStrategy(BaseStrategy[_DummyParams]):
    strategy_id = "dummy"

    @classmethod
    def params_type(cls):
        return _DummyParams

    def indicators(self, data, params):
        return pd.DataFrame(index=data.index)

    def generate_signals(self, data, indicators, ctx, params):
        df = pd.DataFrame({"signal": [0] * len(data), "size": [1.0] * len(data)}, index=data.index)
        return SignalFrame(data=df)

    def warmup_bars(self, params):
        return params.lookback


def _ohlcv(n=10):
    idx = pd.date_range("2024-01-01", periods=n, freq="D")
    return pd.DataFrame(
        {"open": 1.0, "high": 1.0, "low": 1.0, "close": 1.0, "volume": 100},
        index=idx,
    )


def test_cannot_instantiate_base_directly():
    with pytest.raises(TypeError):
        BaseStrategy()  # type: ignore[abstract]


def test_concrete_strategy_instantiable():
    s = _DummyStrategy()
    assert s.strategy_id == "dummy"
    assert s.version == "1.0"


def test_validate_passes_with_required_columns():
    s = _DummyStrategy()
    s.validate(_ohlcv(), _DummyParams())


def test_validate_raises_on_missing_columns():
    s = _DummyStrategy()
    bad = _ohlcv().drop(columns=["volume"])
    with pytest.raises(ValueError, match="volume"):
        s.validate(bad, _DummyParams())


def test_warmup_default_is_zero():
    class _NoWarmup(_DummyStrategy):
        def warmup_bars(self, params):
            return BaseStrategy.warmup_bars(self, params)
    assert _NoWarmup().warmup_bars(_DummyParams()) == 0


def test_indicators_and_signals_callable():
    s = _DummyStrategy()
    data = _ohlcv()
    p = _DummyParams()
    ind = s.indicators(data, p)
    ctx = StrategyContext(symbol="X", timeframe="1d", warmup_bars=s.warmup_bars(p))
    sf = s.generate_signals(data, ind, ctx, p)
    assert isinstance(sf, SignalFrame)
    assert "signal" in sf.data.columns
