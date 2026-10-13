from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class Params:
    memory_bars: int = 64
    fractional_order: float = 0.45
    recent_bars: int = 10
    baseline_bars: int = 40
    correlation_threshold: float = 0.20
    holding_bars: int = 18


class GeneratedStrategy(BaseStrategy[Params]):
    strategy_id = "gen_local_1791251044"

    @classmethod
    def params_type(cls):
        return Params

    def warmup_bars(self, params: Params) -> int:
        return params.memory_bars + params.recent_bars + params.baseline_bars + 2

    def indicators(self, data: pd.DataFrame, params: Params) -> pd.DataFrame:
        if not (params.memory_bars >= 2 and 0 < params.fractional_order < 1
                and params.recent_bars >= 3 and params.baseline_bars >= 3
                and 0 < params.correlation_threshold < 1 and params.holding_bars >= 1):
            raise ValueError("Invalid strategy parameters")
        out = pd.DataFrame(index=data.index)
        close = data["close"].astype(float).where(data["close"] > 0)
        fractional = close.copy()
        weight = 1.0
        for lag in range(1, params.memory_bars):
            weight *= (lag - 1 - params.fractional_order) / lag
            fractional = fractional + weight * close.shift(lag)
        out["fractional_close"] = fractional
        out["innovation"] = fractional.diff().div(close.shift(1)).replace([np.inf, -np.inf], np.nan)
        out["lagged_innovation"] = out["innovation"].shift(1)
        out["recent_correlation"] = out["innovation"].rolling(params.recent_bars).corr(out["lagged_innovation"]).replace([np.inf, -np.inf], np.nan)
        out["baseline_correlation"] = out["innovation"].rolling(params.baseline_bars).corr(out["lagged_innovation"]).shift(params.recent_bars).replace([np.inf, -np.inf], np.nan)
        out["recent_drift"] = out["innovation"].rolling(params.recent_bars).mean()
        out["candidate"] = ((out["baseline_correlation"] < -params.correlation_threshold)
                            & (out["recent_correlation"] > params.correlation_threshold)
                            & (out["recent_drift"] > 0)).fillna(False)
        out["confirmed"] = out["candidate"] & out["candidate"].shift(1, fill_value=False)
        out["entry_event"] = out["confirmed"] & ~out["confirmed"].shift(1, fill_value=False)
        return out

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame,
                         ctx: StrategyContext, params: Params) -> SignalFrame:
        if params.holding_bars < 1:
            raise ValueError("holding_bars must be positive")
        signal = np.zeros(len(data), dtype=int)
        exit_bar = -1
        for bar, entry in enumerate(indicators["entry_event"].fillna(False).to_numpy(dtype=bool)):
            if exit_bar >= 0:
                if bar == exit_bar:
                    exit_bar = -1
                    continue
                signal[bar] = 1
            elif entry:
                exit_bar = bar + params.holding_bars
                signal[bar] = 1
        df = pd.DataFrame({"signal": signal, "size": 0.95}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


if __name__ == "__main__":
    strategy = GeneratedStrategy()
    params = Params()
    index = pd.date_range("2020-01-01", periods=50)
    data = pd.DataFrame({"close": 100.0}, index=index)
    events = pd.DataFrame({"entry_event": True}, index=index)
    result = strategy.generate_signals(data, events, None, params).data
    assert result["signal"].iloc[0] == 0
    assert result["signal"].iloc[1:19].eq(1).all()
    assert result["signal"].iloc[19] == 0
    assert result["signal"].isin([0, 1]).all()
    assert result["size"].gt(0).all()
