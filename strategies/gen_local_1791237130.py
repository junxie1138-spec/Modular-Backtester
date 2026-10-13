from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class CompressionParams:
    drift_bars: int = 18
    compression_bars: int = 8
    baseline_bars: int = 60
    compression_ratio: float = 0.55


class GeneratedStrategy(BaseStrategy[CompressionParams]):
    strategy_id = "gen_local_1791237130"

    @classmethod
    def params_type(cls):
        return CompressionParams

    def warmup_bars(self, params: CompressionParams) -> int:
        return max(200, params.drift_bars + 1,
                   params.compression_bars + 1, params.baseline_bars + 1)

    def indicators(self, data: pd.DataFrame, params: CompressionParams) -> pd.DataFrame:
        close = data["close"].astype(float).where(data["close"] > 0)
        returns = close.pct_change(fill_method=None).replace([np.inf, -np.inf], np.nan)
        drift_window = returns.rolling(params.drift_bars, min_periods=params.drift_bars)
        short_window = returns.rolling(params.compression_bars, min_periods=params.compression_bars)
        base_window = returns.rolling(params.baseline_bars, min_periods=params.baseline_bars)
        short_range = short_window.max() - short_window.min()
        base_range = base_window.max() - base_window.min()
        return pd.DataFrame({
            "returns": returns,
            "trimmed_drift": drift_window.sum() - drift_window.max() - drift_window.min(),
            "short_return_range": short_range,
            "base_return_range": base_range,
            "compression": short_range / base_range.where(base_range > 0),
            "ma200": close.rolling(200, min_periods=200).mean(),
        }, index=data.index)

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame,
                         ctx: StrategyContext, params: CompressionParams) -> SignalFrame:
        valid = indicators[["trimmed_drift", "compression", "ma200"]].notna().all(axis=1)
        drift = indicators["trimmed_drift"]
        direction = pd.Series(np.select([drift > 0, drift < 0], [1, -1], default=0),
                              index=data.index, dtype=int)
        compressed = valid & (indicators["compression"] <= params.compression_ratio)
        long_entry = compressed & (direction == 1) & (data["close"] > indicators["ma200"])
        short_entry = compressed & (direction == -1) & (data["close"] < indicators["ma200"])
        entries = np.select([long_entry, short_entry], [1, -1], default=0)
        positions = np.zeros(len(data), dtype=int)
        position = 0
        for i, entry in enumerate(entries):
            # Only an opposite trimmed-drift direction exits; gates govern new entries.
            if position and direction.iat[i] == -position:
                position = 0
            elif position == 0:
                position = int(entry)
            positions[i] = position
        df = pd.DataFrame({"signal": positions, "size": 0.95}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check():
    index = pd.date_range("2020-01-01", periods=8)
    data = pd.DataFrame({"close": 101.0}, index=index)
    indicators = pd.DataFrame({
        "trimmed_drift": [1, 1, -1, -1, -1, 1, 1, 1],
        "compression": [0.4] * 8,
        "ma200": [100, 100, 102, 102, 102, 100, 100, 100],
    }, index=index)
    strategy = object.__new__(GeneratedStrategy)
    result = strategy.generate_signals(data, indicators, None, CompressionParams())
    assert result.data["signal"].tolist() == [0, 1, 1, 0, -1, -1, 0, 1]
    assert result.data["size"].gt(0).all()


if __name__ == "__main__":
    _self_check()
