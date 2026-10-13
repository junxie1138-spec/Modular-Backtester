from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class ContagionParams:
    graph_window: int = 16
    contagion_threshold: float = 0.25
    holding_bars: int = 8


class GeneratedStrategy(BaseStrategy[ContagionParams]):
    strategy_id = "gen_local_1790940386"

    @classmethod
    def params_type(cls) -> type[ContagionParams]:
        return ContagionParams

    @staticmethod
    def warmup_bars(params: ContagionParams) -> int:
        return max(200, params.graph_window + 1)

    @staticmethod
    def indicators(data: pd.DataFrame, params: ContagionParams) -> pd.DataFrame:
        if params.graph_window < 3 or params.holding_bars < 1:
            raise ValueError("graph_window must be >= 3 and holding_bars >= 1")
        if not 0.0 <= params.contagion_threshold <= 1.0:
            raise ValueError("contagion_threshold must be between zero and one")

        close = data["close"].astype(float)
        returns = np.log(close.where(close > 0)).diff()
        width = params.graph_window
        nodes = np.column_stack([
            returns.shift(width - 1 - offset).to_numpy()
            for offset in range(width)
        ])
        valid = np.isfinite(nodes).all(axis=1)
        edges = np.zeros((len(data), width, width), dtype=bool)

        # Horizontal visibility: both endpoints exceed every intervening return.
        for left in range(width - 1):
            barrier = np.full(len(data), -np.inf)
            for right in range(left + 1, width):
                edges[:, left, right] = (
                    valid
                    & (np.minimum(nodes[:, left], nodes[:, right]) > barrier)
                )
                barrier = np.maximum(barrier, nodes[:, right])

        incoming = edges.sum(axis=1)
        outgoing = edges.sum(axis=2)
        all_paths = (incoming * outgoing).sum(axis=1)
        positive = nodes > 0.0
        positive_edges = edges & positive[:, :, None] & positive[:, None, :]
        positive_paths = (
            positive_edges.sum(axis=1) * positive_edges.sum(axis=2)
        ).sum(axis=1)
        contagion = np.divide(
            positive_paths,
            all_paths,
            out=np.zeros(len(data), dtype=float),
            where=all_paths > 0,
        )
        contagion[~valid] = np.nan

        return pd.DataFrame({
            "log_return": returns,
            "ma_200": close.rolling(200, min_periods=200).mean(),
            "momentum": close / close.shift(width) - 1.0,
            "visibility_paths": np.where(valid, all_paths, np.nan),
            "positive_visibility_paths": np.where(valid, positive_paths, np.nan),
            "contagion": contagion,
        }, index=data.index)

    @staticmethod
    def generate_signals(
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: ContagionParams,
    ) -> SignalFrame:
        eligible = (
            (data["close"] > indicators["ma_200"])
            & (indicators["momentum"] > 0.0)
            & (indicators["contagion"] >= params.contagion_threshold)
        ).fillna(False).to_numpy(dtype=bool)

        signal = np.zeros(len(data), dtype=int)
        exit_bar = -1
        for bar in range(len(data)):
            if exit_bar >= 0:
                if bar == exit_bar:
                    exit_bar = -1
                else:
                    signal[bar] = 1
            elif eligible[bar]:
                signal[bar] = 1
                exit_bar = bar + params.holding_bars

        df = pd.DataFrame({"signal": signal, "size": 0.95}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


if __name__ == "__main__":
    prices = 100.0 + np.arange(240)
    sample = pd.DataFrame({
        "open": prices,
        "high": prices + 1.0,
        "low": prices - 1.0,
        "close": prices,
        "volume": np.full(240, 1000000.0),
    }, index=pd.date_range("2020-01-01", periods=240, freq="B"))
    params = ContagionParams()
    derived = GeneratedStrategy.indicators(sample, params)
    result = GeneratedStrategy.generate_signals(sample, derived, None, params).data
    assert result["signal"].iloc[:200].eq(0).all()
    assert result["signal"].iloc[200:208].eq(1).all()
    assert result["signal"].iloc[208] == 0
    assert result["signal"].isin([0, 1]).all()
    assert result["size"].gt(0.0).all()
