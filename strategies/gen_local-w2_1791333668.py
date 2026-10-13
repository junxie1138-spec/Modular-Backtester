from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class GeneratedParams:
    graph_window: int = 8
    max_bridge_fraction: float = 0.15
    hold_bars: int = 8

    def __post_init__(self):
        if not isinstance(self.graph_window, int) or not 4 <= self.graph_window <= 9:
            raise ValueError("graph_window must be an integer from 4 to 9")
        if not 0.0 <= self.max_bridge_fraction <= 1.0:
            raise ValueError("max_bridge_fraction must be between 0 and 1")
        if not isinstance(self.hold_bars, int) or not 5 <= self.hold_bars <= 10:
            raise ValueError("hold_bars must be an integer from 5 to 10")


class GeneratedStrategy(BaseStrategy[GeneratedParams]):
    strategy_id = "gen_local-w2_1791333668"

    @classmethod
    def params_type(cls):
        return GeneratedParams

    def warmup_bars(self, params: GeneratedParams) -> int:
        return params.graph_window + 1

    def indicators(self, data: pd.DataFrame, params: GeneratedParams) -> pd.DataFrame:
        w = params.graph_window
        volume = data["volume"].astype(float).replace([np.inf, -np.inf], np.nan)
        volumes = np.column_stack([
            volume.shift(lag).to_numpy(dtype=float) for lag in range(w, 0, -1)
        ])
        valid = (
            np.isfinite(volumes).all(axis=1)
            & (volumes >= 0).all(axis=1)
            & (volumes.sum(axis=1) > 0)
        )
        bridges = np.zeros(len(data), dtype=float)
        possible = 0
        # Horizontal visibility: both endpoints exceed every intervening node.
        # Count only nonadjacent edges crossing the window's central cut.
        for left in range(w // 2):
            for right in range(w // 2, w):
                if right - left > 1:
                    visible = np.minimum(volumes[:, left], volumes[:, right]) > volumes[:, left + 1:right].max(axis=1)
                    bridges += visible
                    possible += 1
        out = pd.DataFrame(index=data.index)
        out["bridge_fraction"] = np.where(valid, bridges / possible, np.nan)
        out["upper"] = data["high"].replace([np.inf, -np.inf], np.nan).shift(1).rolling(w, min_periods=w).max()
        out["lower"] = data["low"].replace([np.inf, -np.inf], np.nan).shift(1).rolling(w, min_periods=w).min()
        return out

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: GeneratedParams,
    ) -> SignalFrame:
        close = data["close"].replace([np.inf, -np.inf], np.nan)
        eligible = (
            indicators["bridge_fraction"].le(params.max_bridge_fraction)
            & indicators["upper"].notna()
            & indicators["lower"].notna()
        )
        entries = np.select(
            [eligible & close.gt(indicators["upper"]), eligible & close.lt(indicators["lower"])],
            [1, -1], default=0,
        )
        raw = np.zeros(len(data), dtype=int)
        position = 0
        entry_bar = -1
        for i, entry in enumerate(entries):
            if position:
                if i - entry_bar == params.hold_bars:
                    position = 0
                # Opposite setups cannot reverse or shorten a live trade.
            elif entry:
                position = int(entry)
                entry_bar = i
            raw[i] = position
        df = pd.DataFrame(index=data.index)
        df["signal"] = raw
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = 1.0
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check():
    params = GeneratedParams()
    strategy = GeneratedStrategy()
    index = pd.date_range("2020-01-01", periods=22, freq="D")
    data = pd.DataFrame({
        "open": 100.0, "high": 101.0, "low": 99.0,
        "close": 100.0, "volume": np.arange(1, 23, dtype=float),
    }, index=index)
    derived = strategy.indicators(data, params)
    assert derived["bridge_fraction"].iloc[:8].isna().all()
    assert derived["bridge_fraction"].iloc[8:].eq(0.0).all()
    assert derived.iloc[:12].equals(strategy.indicators(data.iloc[:12], params))
    assert strategy.warmup_bars(GeneratedParams(graph_window=9)) == 10
    data.loc[index[[1, 12]], "close"] = 102.0
    data.loc[index[[2, 11]], "close"] = 98.0
    setups = pd.DataFrame({
        "bridge_fraction": 0.0, "upper": 101.0, "lower": 99.0,
    }, index=index)
    ctx = StrategyContext(symbol="SPY", timeframe="1d", warmup_bars=9)
    actual = strategy.generate_signals(data, setups, ctx, params).data
    expected = np.zeros(22, dtype=int)
    expected[2:10] = 1
    expected[12:20] = -1
    assert np.array_equal(actual["signal"].to_numpy(), expected)
    assert actual["size"].gt(0).all()


if __name__ == "__main__":
    _self_check()
