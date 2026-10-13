from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class CompressionParams:
    range_window: int = 8
    roc_window: int = 5
    rank_window: int = 80
    compression_quantile: float = 0.25
    acceleration_quantile: float = 0.85


class GeneratedStrategy(BaseStrategy[CompressionParams]):
    strategy_id = "gen_local-w2_1791336768"

    @classmethod
    def params_type(cls):
        return CompressionParams

    def warmup_bars(self, params: CompressionParams) -> int:
        return int(max(params.range_window + params.rank_window + 5,
                       params.roc_window + params.rank_window + 4))

    def indicators(self, data: pd.DataFrame, params: CompressionParams) -> pd.DataFrame:
        if min(params.range_window, params.roc_window, params.rank_window) < 2:
            raise ValueError("Lookbacks must be at least two bars")
        if not 0.0 < params.compression_quantile < 1.0:
            raise ValueError("compression_quantile must lie between zero and one")
        if not 0.5 < params.acceleration_quantile < 1.0:
            raise ValueError("acceleration_quantile must lie between one-half and one")

        out = pd.DataFrame(index=data.index)
        close = data["close"].where(data["close"] > 0)
        out["range_high"] = data["high"].rolling(params.range_window).max()
        out["range_low"] = data["low"].rolling(params.range_window).min()
        out["envelope"] = ((out["range_high"] - out["range_low"]) / close).replace([np.inf, -np.inf], np.nan)
        out["compression_threshold"] = out["envelope"].shift(1).rolling(params.rank_window).quantile(params.compression_quantile)
        out["compressed"] = (out["envelope"].le(out["compression_threshold"]) & out["compression_threshold"].notna()).astype(int)
        out["compression_count"] = out["compressed"].rolling(5).sum()
        out["roc"] = (close / close.shift(params.roc_window) - 1.0).replace([np.inf, -np.inf], np.nan)
        out["acceleration"] = out["roc"].diff().rolling(3).mean()
        out["release_pressure"] = (out["acceleration"] / out["envelope"].where(out["envelope"] > 0)).replace([np.inf, -np.inf], np.nan)
        history = out["release_pressure"].shift(1).rolling(params.rank_window)
        out["upper_threshold"] = history.quantile(params.acceleration_quantile)
        out["lower_threshold"] = history.quantile(1.0 - params.acceleration_quantile)
        ready = out[["compression_count", "release_pressure", "upper_threshold", "lower_threshold"]].notna().all(axis=1)
        compressed_recently = out["compression_count"].ge(3)
        long_entry = ready & compressed_recently & out["release_pressure"].gt(0) & out["release_pressure"].gt(out["upper_threshold"])
        short_entry = ready & compressed_recently & out["release_pressure"].lt(0) & out["release_pressure"].lt(out["lower_threshold"])
        out["entry_direction"] = np.select([long_entry, short_entry], [1, -1], default=0).astype(int)
        return out

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame,
                         ctx: StrategyContext, params: CompressionParams) -> SignalFrame:
        entries = indicators["entry_direction"].fillna(0).to_numpy(dtype=int)
        positions = np.zeros(len(data), dtype=int)
        position = 0
        for i, entry in enumerate(entries):
            if position == 0:
                position = int(entry)
            elif entry == -position:
                position = 0
            positions[i] = position

        df = pd.DataFrame(index=data.index)
        df["signal"] = positions
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = 0.95
        return SignalFrame(data=df, signal_column="signal", size_column="size")


if __name__ == "__main__":
    sample = pd.DataFrame(index=pd.date_range("2020-01-01", periods=7))
    entries = pd.DataFrame({"entry_direction": [1, 0, -1, -1, 0, 1, 1]}, index=sample.index)
    strategy = GeneratedStrategy.__new__(GeneratedStrategy)
    result = strategy.generate_signals(sample, entries, None, CompressionParams())
    assert result.data["signal"].tolist() == [0, 1, 1, 0, -1, -1, 0]
