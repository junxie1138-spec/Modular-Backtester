from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class CompressionParams:
    pocket_bars: int = 10
    rank_bars: int = 80
    compression_cutoff: float = 0.45
    pressure_min: float = 0.10


class GeneratedStrategy(BaseStrategy[CompressionParams]):
    strategy_id = "gen_local-w2_1791207887"

    @classmethod
    def params_type(cls):
        return CompressionParams

    @staticmethod
    def warmup_bars(params: CompressionParams) -> int:
        return params.pocket_bars + params.rank_bars - 1

    @staticmethod
    def indicators(data: pd.DataFrame, params: CompressionParams) -> pd.DataFrame:
        df = pd.DataFrame(index=data.index)
        n = params.pocket_bars
        df["daily_range"] = (data["high"] - data["low"]).clip(lower=0)
        df["range_sum"] = df["daily_range"].rolling(n, min_periods=n).sum()
        df["envelope_width"] = (
            data["high"].rolling(n, min_periods=n).max()
            - data["low"].rolling(n, min_periods=n).min()
        )
        denominator = df["range_sum"].replace(0, np.nan)
        df["packing_ratio"] = df["envelope_width"] / denominator
        df["compression_rank"] = df["packing_ratio"].rolling(
            params.rank_bars, min_periods=params.rank_bars
        ).rank(pct=True)
        df["body"] = data["close"] - data["open"]
        df["body_sum"] = df["body"].rolling(n, min_periods=n).sum()
        df["pressure"] = (df["body_sum"] / denominator).clip(-1, 1)
        df["compression_strength"] = (
            (params.compression_cutoff - df["compression_rank"])
            / params.compression_cutoff
        ).clip(0, 1)
        df["pressure_strength"] = (
            (df["pressure"] - params.pressure_min) / (1 - params.pressure_min)
        ).clip(0, 1)
        return df.replace([np.inf, -np.inf], np.nan)

    @staticmethod
    def generate_signals(
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: CompressionParams,
    ) -> SignalFrame:
        # The entry predicate turning false is the only exit.
        entry = (
            indicators["compression_rank"].le(params.compression_cutoff)
            & indicators["pressure"].gt(params.pressure_min)
        ).fillna(False)
        strength = (
            indicators["compression_strength"] * indicators["pressure_strength"]
        ).fillna(0).clip(0, 1)
        df = pd.DataFrame(index=data.index)
        df["signal"] = entry.astype(int)
        df["size"] = 0.25 + 0.70 * strength
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = df["size"].shift(1).fillna(0.25).astype(float)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check():
    params = CompressionParams()
    index = pd.date_range("2020-01-01", periods=5)
    data = pd.DataFrame(index=index)
    features = pd.DataFrame({
        "compression_rank": [np.nan, 0.10, 0.20, 0.80, 0.10],
        "pressure": [np.nan, 0.30, 0.30, 0.30, -0.30],
        "compression_strength": [np.nan, 0.80, 0.40, 0.00, 0.80],
        "pressure_strength": [np.nan, 0.50, 0.50, 0.50, 0.00],
    }, index=index)
    result = GeneratedStrategy.generate_signals(data, features, None, params).data
    assert result["signal"].tolist() == [0, 0, 1, 1, 0]
    assert np.allclose(result["size"], [0.25, 0.25, 0.53, 0.39, 0.25])
    assert result["size"].gt(0).all()


if __name__ == "__main__":
    _self_check()
