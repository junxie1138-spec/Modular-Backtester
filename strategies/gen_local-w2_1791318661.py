from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class GapMemoryParams:
    correlation_window: int = 40
    forecast_window: int = 4
    entry_threshold: float = 0.12
    size_floor: float = 0.20
    size_cap: float = 0.95
    size_gain: float = 0.50


class GeneratedStrategy(BaseStrategy[GapMemoryParams]):
    strategy_id = "gen_local-w2_1791318661"

    @classmethod
    def params_type(cls) -> type[GapMemoryParams]:
        return GapMemoryParams

    def warmup_bars(self, params: GapMemoryParams) -> int:
        return int(params.correlation_window + params.forecast_window + 2)

    def indicators(self, data: pd.DataFrame, params: GapMemoryParams) -> pd.DataFrame:
        if params.correlation_window < 3 or params.forecast_window < 1:
            raise ValueError("Lookback windows must be positive; correlation requires at least three bars.")
        values = (params.entry_threshold, params.size_floor, params.size_cap, params.size_gain)
        if not all(np.isfinite(value) for value in values):
            raise ValueError("Threshold and sizing parameters must be finite.")
        if params.entry_threshold <= 0 or params.size_gain <= 0:
            raise ValueError("Entry threshold and sizing gain must be positive.")
        if not 0 < params.size_floor <= params.size_cap <= 1:
            raise ValueError("Sizing requires 0 < size_floor <= size_cap <= 1.")
        out = pd.DataFrame(index=data.index)
        out["prior_close"] = data["close"].shift(1).replace(0, np.nan)
        out["gap"] = (data["open"] / out["prior_close"] - 1).replace([np.inf, -np.inf], np.nan)
        out["lag_gap"] = out["gap"].shift(1)
        window = params.correlation_window
        out["gap_mean"] = out["gap"].rolling(window, min_periods=window).mean().shift(1)
        out["gap_std"] = out["gap"].rolling(window, min_periods=window).std().shift(1).replace(0, np.nan)
        out["autocorrelation"] = out["gap"].rolling(window, min_periods=window).corr(out["lag_gap"]).shift(1).clip(-1, 1)
        out["gap_innovation"] = (out["gap"] - out["gap_mean"]) / out["gap_std"]
        out["forecast"] = out["autocorrelation"] * out["gap_innovation"]
        out["score"] = out["forecast"].rolling(params.forecast_window, min_periods=params.forecast_window).mean()
        return out.replace([np.inf, -np.inf], np.nan)

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: GapMemoryParams) -> SignalFrame:
        score = indicators["score"].reindex(data.index).replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy()
        signals = np.zeros(len(data), dtype=int)
        sizes = np.full(len(data), params.size_floor, dtype=float)
        state = 0
        held_size = params.size_floor
        for i, value in enumerate(score):
            direction = 1 if value >= params.entry_threshold else -1 if value <= -params.entry_threshold else 0
            if state and direction == -state:
                state = 0
                held_size = params.size_floor
            elif state == 0 and direction:
                state = direction
                held_size = float(np.clip(params.size_floor + params.size_gain * abs(value), params.size_floor, params.size_cap))
            signals[i] = state
            sizes[i] = held_size
        df = pd.DataFrame({"signal": signals, "size": sizes}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = df["size"].shift(1).fillna(params.size_floor).astype(float)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


if __name__ == "__main__":
    strategy = GeneratedStrategy()
    params = GapMemoryParams()
    index = pd.date_range("2020-01-01", periods=8)
    data = pd.DataFrame({"open": 100.0, "high": 101.0, "low": 99.0, "close": 100.0, "volume": 1000.0}, index=index)
    indicators = pd.DataFrame({"score": [np.nan, 0.4, 0.2, 0.0, -0.4, -0.4, 0.0, 0.4]}, index=index)
    result = strategy.generate_signals(data, indicators, None, params).data
    assert result["signal"].tolist() == [0, 0, 1, 1, 1, 0, -1, -1]
    assert result["size"].gt(0).all()
    assert np.isclose(result["size"].iloc[2], 0.4)
    assert np.isclose(result["size"].iloc[4], 0.4)
    stronger = indicators.copy()
    stronger.iloc[1, 0] = 0.8
    stronger_result = strategy.generate_signals(data, stronger, None, params).data
    assert stronger_result["size"].iloc[2] > result["size"].iloc[2]
    flat_index = pd.date_range("2020-01-01", periods=100)
    flat = data.iloc[:1].reindex(flat_index).ffill()
    flat_indicators = strategy.indicators(flat, params)
    assert strategy.generate_signals(flat, flat_indicators, None, params).data["signal"].eq(0).all()
