from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class CompressionParams:
    compression_window: int = 4
    baseline_window: int = 12
    efficiency_window: int = 4
    compression_ratio: float = 0.75
    minimum_efficiency: float = 0.25


class GeneratedStrategy(BaseStrategy[CompressionParams]):
    strategy_id = "gen_local_1791292464"

    @classmethod
    def params_type(cls) -> type[CompressionParams]:
        return CompressionParams

    @staticmethod
    def warmup_bars(params: CompressionParams) -> int:
        return int(max(
            params.compression_window + params.baseline_window,
            params.efficiency_window + 1,
        ))

    @staticmethod
    def indicators(data: pd.DataFrame, params: CompressionParams) -> pd.DataFrame:
        if min(params.compression_window, params.baseline_window, params.efficiency_window) < 1:
            raise ValueError("Window lengths must be positive")
        if not 0.0 < params.compression_ratio < 1.0:
            raise ValueError("compression_ratio must be between zero and one")
        if not 0.0 < params.minimum_efficiency <= 1.0:
            raise ValueError("minimum_efficiency must be positive and at most one")

        out = pd.DataFrame(index=data.index)
        close = data["close"].where(data["close"] > 0)
        span = (data["high"] - data["low"]).where(data["high"] >= data["low"])
        out["normalized_range"] = span / close
        out["recent_range"] = out["normalized_range"].rolling(
            params.compression_window, min_periods=params.compression_window
        ).mean()
        out["baseline_range"] = out["normalized_range"].shift(
            params.compression_window
        ).rolling(params.baseline_window, min_periods=params.baseline_window).mean()
        out["range_ratio"] = out["recent_range"] / out["baseline_range"].where(
            out["baseline_range"] > 0
        )
        out["close_change"] = close.diff()
        out["net_movement"] = out["close_change"].rolling(
            params.efficiency_window, min_periods=params.efficiency_window
        ).sum()
        out["total_movement"] = out["close_change"].abs().rolling(
            params.efficiency_window, min_periods=params.efficiency_window
        ).sum()
        out["directional_efficiency"] = out["net_movement"] / out["total_movement"].where(
            out["total_movement"] > 0
        )
        return out.replace([np.inf, -np.inf], np.nan)

    @staticmethod
    def generate_signals(
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: CompressionParams,
    ) -> SignalFrame:
        compressed = indicators["range_ratio"].lt(params.compression_ratio)
        advancing = indicators["directional_efficiency"].gt(params.minimum_efficiency)
        valid = indicators[["range_ratio", "directional_efficiency"]].notna().all(axis=1)
        # The position exits whenever the two-primitive entry condition flips false.
        df = pd.DataFrame(index=data.index)
        df["signal"] = (compressed & advancing & valid).astype(int)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = 0.95
        return SignalFrame(data=df, signal_column="signal", size_column="size")


if __name__ == "__main__":
    index = pd.date_range("2020-01-01", periods=6)
    data = pd.DataFrame(index=index)
    evidence = pd.DataFrame({
        "range_ratio": [np.nan, 0.6, 0.6, 0.6, 0.9, 0.6],
        "directional_efficiency": [0.5, 0.5, 0.5, -0.1, 0.5, 0.5],
    }, index=index)
    result = GeneratedStrategy.generate_signals(data, evidence, None, CompressionParams())
    assert result.data["signal"].tolist() == [0, 0, 1, 1, 0, 0]
    assert result.data["size"].gt(0).all()
    assert GeneratedStrategy.warmup_bars(CompressionParams()) == 16
