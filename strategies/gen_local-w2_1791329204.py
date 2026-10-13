from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class RangeAnalogParams:
    block_bars: int = 5
    horizon_bars: int = 20
    history_bars: int = 252
    neighbors: int = 7
    reversal_threshold: float = 0.002


class GeneratedStrategy(BaseStrategy[RangeAnalogParams]):
    strategy_id = "gen_local-w2_1791329204"

    @classmethod
    def params_type(cls):
        return RangeAnalogParams

    @staticmethod
    def warmup_bars(params):
        return int(params.history_bars + 4 * params.block_bars)

    @staticmethod
    def indicators(data, params):
        if params.block_bars < 1 or params.horizon_bars < 1:
            raise ValueError("Block and horizon lengths must be positive")
        if params.history_bars <= params.horizon_bars:
            raise ValueError("History must exceed the forecast horizon")
        if not 1 <= params.neighbors <= params.history_bars - params.horizon_bars:
            raise ValueError("Invalid neighbor count")
        if not np.isfinite(params.reversal_threshold) or params.reversal_threshold < 0:
            raise ValueError("Threshold must be finite and nonnegative")

        out = pd.DataFrame(index=data.index)
        close = data["close"].where(data["close"] > 0)
        out["relative_range"] = ((data["high"] - data["low"]).clip(lower=0) / close).replace([np.inf, -np.inf], np.nan)
        block = out["relative_range"].rolling(params.block_bars, min_periods=params.block_bars).mean()
        envelope = out["relative_range"].rolling(4 * params.block_bars, min_periods=4 * params.block_bars).mean().replace(0, np.nan)
        columns = []
        for j in range(4):
            name = f"range_lobe_{j}"
            out[name] = block.shift(j * params.block_bars) / envelope - 1.0
            columns.append(name)

        shape = out[columns]
        distances = []
        outcomes = []
        for lag in range(params.horizon_bars + 1, params.history_bars + 1):
            distances.append(((shape - shape.shift(lag)) ** 2).sum(axis=1, min_count=4).to_numpy())
            outcomes.append((close.shift(lag - params.horizon_bars) / close.shift(lag) - 1.0).to_numpy())

        distance = np.column_stack(distances)
        outcome = np.column_stack(outcomes)
        valid = np.isfinite(distance) & np.isfinite(outcome)
        safe_distance = np.where(valid, distance, np.inf)
        nearest = np.argpartition(safe_distance, params.neighbors - 1, axis=1)[:, :params.neighbors]
        selected_distance = np.take_along_axis(safe_distance, nearest, axis=1)
        selected_outcome = np.take_along_axis(outcome, nearest, axis=1)
        weights = np.where(np.isfinite(selected_distance), 1.0 / (0.05 + selected_distance), 0.0)
        numerator = (weights * np.where(np.isfinite(selected_outcome), selected_outcome, 0.0)).sum(axis=1)
        denominator = weights.sum(axis=1)
        analog = np.divide(numerator, denominator, out=np.zeros(len(data)), where=denominator > 0)
        count = valid.sum(axis=1)
        baseline = np.divide(np.where(valid, outcome, 0.0).sum(axis=1), count, out=np.zeros(len(data)), where=count > 0)
        ready = valid.all(axis=1)
        out["ready"] = ready
        out["analog_return"] = np.where(ready, analog, 0.0)
        out["baseline_return"] = np.where(ready, baseline, 0.0)
        out["forecast"] = np.where(ready, analog - baseline, 0.0)
        return out

    @staticmethod
    def generate_signals(data, indicators, ctx, params):
        forecast = indicators["forecast"].replace([np.inf, -np.inf], np.nan)
        ready = indicators["ready"].fillna(False).astype(bool)
        events = pd.Series(np.nan, index=data.index, dtype=float)
        events.loc[ready & (forecast > params.reversal_threshold)] = 1.0
        events.loc[ready & (forecast < -params.reversal_threshold)] = 0.0
        df = pd.DataFrame(index=data.index)
        df["signal"] = events.ffill().fillna(0).astype(int)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = 0.95
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check():
    params = RangeAnalogParams()
    index = pd.date_range("2020-01-01", periods=400, freq="B")
    t = np.arange(len(index), dtype=float)
    close = 100.0 * np.exp(0.0003 * t + 0.015 * np.sin(t / 11.0))
    width = close * (0.008 + 0.004 * (1.0 + np.sin(t / 7.0)))
    data = pd.DataFrame({"open": close, "high": close + width, "low": close - width, "close": close, "volume": 1000000.0}, index=index)
    full = GeneratedStrategy.indicators(data, params)
    prefix = GeneratedStrategy.indicators(data.iloc[:330], params)
    np.testing.assert_allclose(full["forecast"].iloc[:330], prefix["forecast"])
    assert not full["ready"].iloc[:GeneratedStrategy.warmup_bars(params) - 1].any()
    probe = pd.DataFrame({"forecast": [-0.01, 0.01, 0.01, 0.0, -0.01, 0.01], "ready": True}, index=index[:6])
    result = GeneratedStrategy.generate_signals(data.iloc[:6], probe, None, params).data
    assert result["signal"].tolist() == [0, 0, 1, 1, 1, 0]
    assert (result["size"] > 0).all()


if __name__ == "__main__":
    _self_check()
