from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class InventoryRankParams:
    inventory_window: int = 20
    rank_window: int = 84
    smoothing_window: int = 18
    entry_threshold: float = 0.12
    minimum_size: float = 0.20
    maximum_size: float = 0.95


class GeneratedStrategy(BaseStrategy[InventoryRankParams]):
    strategy_id = "gen_local_1791308615"

    @classmethod
    def params_type(cls) -> type[InventoryRankParams]:
        return InventoryRankParams

    @staticmethod
    def warmup_bars(params: InventoryRankParams) -> int:
        windows = (params.inventory_window, params.rank_window, params.smoothing_window)
        if any(w < 2 for w in windows):
            raise ValueError("Lookback windows must be at least two bars")
        if not 0.0 < params.entry_threshold < 1.0:
            raise ValueError("entry_threshold must be between zero and one")
        if not 0.0 < params.minimum_size <= params.maximum_size <= 1.0:
            raise ValueError("Sizes must satisfy 0 < minimum <= maximum <= 1")
        return int(sum(windows))

    @staticmethod
    def indicators(data: pd.DataFrame, params: InventoryRankParams) -> pd.DataFrame:
        GeneratedStrategy.warmup_bars(params)
        out = pd.DataFrame(index=data.index)
        out["clean_close"] = data["close"].where(np.isfinite(data["close"]) & (data["close"] > 0))
        out["clean_volume"] = data["volume"].where(np.isfinite(data["volume"]) & (data["volume"] >= 0))
        out["dollar_volume"] = out["clean_close"] * out["clean_volume"]
        out["inventory_value"] = out["dollar_volume"].rolling(params.inventory_window).sum()
        out["inventory_volume"] = out["clean_volume"].rolling(params.inventory_window).sum()
        out["inventory_center"] = out["inventory_value"] / out["inventory_volume"].where(out["inventory_volume"] > 0)
        out["price_percentile"] = out["clean_close"].rolling(params.rank_window).rank(method="average", pct=True)
        out["inventory_percentile"] = out["inventory_center"].rolling(params.rank_window).rank(method="average", pct=True)
        out["rank_displacement"] = out["price_percentile"] - out["inventory_percentile"]
        out["edge"] = out["rank_displacement"].rolling(params.smoothing_window).mean()
        return out.replace([np.inf, -np.inf], np.nan)

    @staticmethod
    def generate_signals(data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: InventoryRankParams) -> SignalFrame:
        GeneratedStrategy.warmup_bars(params)
        edge = indicators["edge"].reindex(data.index).replace([np.inf, -np.inf], np.nan)
        direction = np.select(
            [edge >= params.entry_threshold, edge <= -params.entry_threshold],
            [1, -1],
            default=0,
        )
        signals = np.zeros(len(data), dtype=int)
        position = 0
        for i, target in enumerate(direction):
            if position and target == -position:
                position = 0
            elif position == 0 and target:
                position = int(target)
            signals[i] = position
        conviction = edge.abs().div(1.0 - params.entry_threshold).clip(0.0, 1.0).fillna(0.0)
        df = pd.DataFrame(index=data.index)
        df["signal"] = signals
        df["size"] = params.minimum_size + (params.maximum_size - params.minimum_size) * conviction
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = df["size"].shift(1).fillna(params.minimum_size).astype(float)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check() -> None:
    params = InventoryRankParams()
    index = pd.date_range("2020-01-01", periods=160, freq="B")
    close = 100.0 + np.arange(160) * 0.05 + np.sin(np.arange(160) / 5.0)
    data = pd.DataFrame({"open": close, "high": close + 1.0, "low": close - 1.0, "close": close, "volume": 1000.0}, index=index)
    derived = GeneratedStrategy.indicators(data, params)
    assert derived["edge"].iloc[:119].isna().all()
    assert pd.notna(derived["edge"].iloc[119])
    assert GeneratedStrategy.warmup_bars(params) >= 120
    sample = data.iloc[:8]
    edges = pd.DataFrame({"edge": [np.nan, 0.20, 0.01, -0.20, -0.20, 0.0, 0.20, 0.20]}, index=sample.index)
    result = GeneratedStrategy.generate_signals(sample, edges, None, params).data
    assert result["signal"].tolist() == [0, 0, 1, 1, 0, -1, -1, 0]
    assert result["size"].between(params.minimum_size, params.maximum_size).all()
    assert result["size"].iloc[2] > result["size"].iloc[3]


if __name__ == "__main__":
    _self_check()
