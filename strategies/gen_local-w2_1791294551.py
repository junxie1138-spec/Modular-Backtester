from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class Params:
    fractional_order: float = 0.45
    fractional_window: int = 24
    breakout_window: int = 20
    compression_window: int = 8
    baseline_window: int = 40
    compression_limit: float = 3.0
    holding_bars: int = 18


class GeneratedStrategy(BaseStrategy[Params]):
    strategy_id = "gen_local-w2_1791294551"

    @classmethod
    def params_type(cls) -> type[Params]:
        return Params

    @staticmethod
    def warmup_bars(params: Params) -> int:
        if not 0.0 < params.fractional_order < 1.0:
            raise ValueError("fractional_order must be between zero and one")
        windows = (params.fractional_window, params.breakout_window,
                   params.compression_window, params.baseline_window)
        if any(window < 2 for window in windows):
            raise ValueError("Indicator windows must be at least two")
        if params.holding_bars < 1 or params.compression_limit <= 0:
            raise ValueError("Holding period and compression limit must be positive")
        return int(max(params.fractional_window + params.breakout_window + 1,
                       params.baseline_window + 2,
                       params.compression_window + 1))

    def indicators(self, data: pd.DataFrame, params: Params) -> pd.DataFrame:
        self.warmup_bars(params)
        close = data["close"].astype(float).replace([np.inf, -np.inf], np.nan)
        weights = np.ones(params.fractional_window, dtype=float)
        for k in range(1, params.fractional_window):
            weights[k] = -weights[k - 1] * (params.fractional_order - k + 1) / k
        fractional_close = close * weights[0]
        for k in range(1, params.fractional_window):
            fractional_close = fractional_close + weights[k] * close.shift(k)
        previous_close = close.shift(1)
        true_range = pd.concat([
            data["high"] - data["low"],
            (data["high"] - previous_close).abs(),
            (data["low"] - previous_close).abs(),
        ], axis=1).max(axis=1, skipna=False)
        box_high = data["high"].rolling(params.compression_window).max()
        box_low = data["low"].rolling(params.compression_window).min()
        baseline_range = true_range.rolling(params.baseline_window).mean()
        out = pd.DataFrame(index=data.index)
        out["fractional_close"] = fractional_close
        out["fractional_high"] = fractional_close.shift(1).rolling(params.breakout_window).max()
        out["fractional_low"] = fractional_close.shift(1).rolling(params.breakout_window).min()
        out["true_range"] = true_range
        out["box_high"] = box_high
        out["box_low"] = box_low
        out["baseline_range"] = baseline_range
        out["compression"] = (box_high - box_low) / baseline_range.where(baseline_range > 0)
        return out.replace([np.inf, -np.inf], np.nan)

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame,
                         ctx: StrategyContext, params: Params) -> SignalFrame:
        self.warmup_bars(params)
        columns = ["fractional_close", "fractional_high", "fractional_low", "compression"]
        ready = np.isfinite(indicators[columns]).all(axis=1)
        compressed = ready & indicators["compression"].le(params.compression_limit)
        upward = compressed & indicators["fractional_close"].gt(indicators["fractional_high"])
        downward = compressed & indicators["fractional_close"].lt(indicators["fractional_low"])
        candidate = pd.Series(0, index=data.index, dtype=int)
        candidate.loc[upward] = 1
        candidate.loc[downward] = -1
        confirmed = candidate.where(candidate.eq(candidate.shift(1)), 0)
        signals = np.zeros(len(data), dtype=int)
        position = 0
        entered = -1
        for i, direction in enumerate(confirmed.to_numpy()):
            if position:
                if i - entered >= params.holding_bars:
                    position = 0
            elif direction:
                position = int(direction)
                entered = i
            signals[i] = position
        df = pd.DataFrame(index=data.index)
        df["signal"] = signals
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = 0.95
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check() -> None:
    index = pd.date_range("2020-01-01", periods=14, freq="B")
    data = pd.DataFrame(1.0, index=index, columns=["open", "high", "low", "close", "volume"])
    directions = np.array([0, 1, 1, 0, -1, 0, 0, -1, -1, 0, 1, 0, 0, 0])
    derived = pd.DataFrame({
        "fractional_close": directions.astype(float),
        "fractional_high": 0.5,
        "fractional_low": -0.5,
        "compression": 1.0,
    }, index=index)
    derived.loc[index[0], "compression"] = np.nan
    result = GeneratedStrategy().generate_signals(data, derived, None, Params(holding_bars=3))
    assert result.data["signal"].tolist() == [0, 0, 0, 1, 1, 1, 0, 0, 0, -1, -1, -1, 0, 0]
    assert result.data["size"].gt(0).all()


if __name__ == "__main__":
    _self_check()
