from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class Params:
    fast_bars: int = 5
    slow_bars: int = 40
    compression: float = 0.8
    hold_bars: int = 8


class GeneratedStrategy(BaseStrategy[Params]):
    strategy_id = "gen_local_1790853455"

    @classmethod
    def params_type(cls):
        return Params

    def warmup_bars(self, params: Params) -> int:
        return max(params.fast_bars, params.slow_bars) + 2

    def indicators(self, data: pd.DataFrame, params: Params) -> pd.DataFrame:
        if not (1 <= params.fast_bars < params.slow_bars):
            raise ValueError("Require 1 <= fast_bars < slow_bars")
        if not (0.0 < params.compression <= 1.0 and 5 <= params.hold_bars <= 10):
            raise ValueError("Require 0 < compression <= 1 and 5 <= hold_bars <= 10")
        out = pd.DataFrame(index=data.index)
        out["previous_close"] = data["close"].shift(1)
        out["daily_range"] = data["high"] - data["low"]
        out["high_gap"] = (data["high"] - out["previous_close"]).abs()
        out["low_gap"] = (data["low"] - out["previous_close"]).abs()
        out["true_range"] = out[["daily_range", "high_gap", "low_gap"]].max(axis=1, skipna=False)
        out["fast_atr"] = out["true_range"].rolling(params.fast_bars, min_periods=params.fast_bars).mean()
        out["slow_atr"] = out["true_range"].rolling(params.slow_bars, min_periods=params.slow_bars).mean()
        out["atr_ratio"] = out["fast_atr"] / out["slow_atr"].where(out["slow_atr"] > 0)
        out["month_boundary"] = (data.index.day >= 25) | (data.index.day <= 3)
        out["setup"] = out["month_boundary"] & out["atr_ratio"].le(params.compression) & np.isfinite(out["atr_ratio"])
        out["confirmed"] = out["setup"] & out["setup"].shift(1, fill_value=False)
        return out

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: Params) -> SignalFrame:
        if not 5 <= params.hold_bars <= 10:
            raise ValueError("hold_bars must be between 5 and 10")
        signals = np.zeros(len(data), dtype=int)
        entry_bar = -1
        ready = indicators["confirmed"].fillna(False).to_numpy(dtype=bool)
        for i in range(len(data)):
            if entry_bar >= 0:
                if i - entry_bar == params.hold_bars:
                    entry_bar = -1
                else:
                    signals[i] = 1
            elif ready[i]:
                entry_bar = i
                signals[i] = 1
        df = pd.DataFrame({"signal": signals, "size": 1.0}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check() -> None:
    strategy = GeneratedStrategy()
    params = Params()
    index = pd.bdate_range("2020-01-01", periods=100)
    data = pd.DataFrame({"open": 100.0, "high": 101.0, "low": 99.0, "close": 100.0, "volume": 1000.0}, index=index)
    ind = strategy.indicators(data, params)
    assert not ind["confirmed"].iloc[:strategy.warmup_bars(params) - 1].any()
    assert ind["confirmed"].equals(ind["setup"] & ind["setup"].shift(1, fill_value=False))
    ctx = StrategyContext(symbol="SPY", timeframe="1d", warmup_bars=strategy.warmup_bars(params))
    for persistent in (False, True):
        ready = pd.DataFrame({"confirmed": False}, index=index)
        ready.iloc[2:, 0] = persistent
        ready.iloc[2, 0] = True
        result = strategy.generate_signals(data, ready, ctx, params).data
        assert result["signal"].iloc[:3].eq(0).all()
        assert result["signal"].iloc[3:3 + params.hold_bars].eq(1).all()
        assert result["signal"].iloc[3 + params.hold_bars] == 0
        assert result["signal"].isin([0, 1]).all()
        assert result["size"].gt(0).all()
        if not persistent:
            assert result["signal"].iloc[3 + params.hold_bars:].eq(0).all()


if __name__ == "__main__":
    _self_check()
