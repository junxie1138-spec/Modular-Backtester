from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class Params:
    graph_window: int = 6
    min_compression: float = 0.4
    min_signal_noise: float = 0.5


class GeneratedStrategy(BaseStrategy[Params]):
    strategy_id = "gen_local_1790879121"

    @classmethod
    def params_type(cls):
        return Params

    def warmup_bars(self, params: Params) -> int:
        return max(params.graph_window, 3) + 1

    def indicators(self, data: pd.DataFrame, params: Params) -> pd.DataFrame:
        if params.graph_window < 3:
            raise ValueError("graph_window must be at least 3")
        if not 0.0 < params.min_compression < 1.0:
            raise ValueError("min_compression must be between zero and one")
        if not 0.0 < params.min_signal_noise <= 1.0:
            raise ValueError("min_signal_noise must be between zero and one")
        width = params.graph_window
        volume = data["volume"].astype(float)
        values = np.column_stack([
            volume.shift(width - 1 - j).to_numpy() for j in range(width)
        ])
        valid = np.isfinite(values).all(axis=1) & (values > 0).all(axis=1)
        hops = np.full(values.shape, width + 1, dtype=int)
        hops[:, 0] = 0
        # Directed horizontal visibility: endpoints exceed every intervening node.
        for j in range(1, width):
            barrier = np.full(len(data), -np.inf)
            for i in range(j - 1, -1, -1):
                visible = np.minimum(values[:, i], values[:, j]) > barrier
                hops[:, j] = np.minimum(
                    hops[:, j], np.where(visible, hops[:, i] + 1, width + 1)
                )
                barrier = np.maximum(barrier, values[:, i])
        out = pd.DataFrame(index=data.index)
        out["volume_path_hops"] = np.where(valid, hops[:, -1], np.nan)
        out["compression"] = 1.0 - out["volume_path_hops"] / (width - 1)
        out["return"] = data["close"].astype(float).pct_change(fill_method=None)
        out["return"] = out["return"].replace([np.inf, -np.inf], np.nan)
        out["drift"] = out["return"].rolling(2, min_periods=2).sum()
        out["noise"] = out["return"].abs().rolling(2, min_periods=2).sum()
        out["signal_noise"] = out["drift"] / out["noise"].replace(0.0, np.nan)
        return out

    def generate_signals(
        self, data: pd.DataFrame, indicators: pd.DataFrame,
        ctx: StrategyContext, params: Params,
    ) -> SignalFrame:
        active = indicators["compression"].ge(params.min_compression)
        raw = pd.Series(0, index=data.index, dtype=int)
        raw.loc[active & indicators["signal_noise"].ge(params.min_signal_noise)] = 1
        raw.loc[active & indicators["signal_noise"].le(-params.min_signal_noise)] = -1
        confirmed = raw.where(raw.eq(raw.shift(1)), 0).to_numpy(dtype=int)
        positions = np.zeros(len(data), dtype=int)
        position = 0
        for i, vote in enumerate(confirmed):
            if position and vote == -position:
                position = 0
            elif position == 0 and vote:
                position = int(vote)
            positions[i] = position
        df = pd.DataFrame({"signal": positions, "size": 0.95}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


if __name__ == "__main__":
    strategy = GeneratedStrategy()
    params = Params()
    index = pd.date_range("2020-01-01", periods=10)
    close = np.arange(100.0, 110.0)
    data = pd.DataFrame({
        "open": close, "high": close + 1, "low": close - 1,
        "close": close, "volume": [10, 1, 2, 3, 4, 10, 1, 1, 1, 1],
    }, index=index)
    calculated = strategy.indicators(data, params)
    assert np.isclose(calculated["compression"].iloc[5], 0.8)
    assert calculated["compression"].iloc[:5].isna().all()
    calculated["compression"] = 1.0
    calculated["signal_noise"] = [0, .9, .9, 0, -.9, -.9, -.9, .9, .9, .9]
    ctx = StrategyContext("SPY", "1d", strategy.warmup_bars(params))
    result = strategy.generate_signals(data, calculated, ctx, params).data
    assert result["signal"].tolist() == [0, 0, 0, 1, 1, 1, 0, -1, -1, 0]
    assert result["size"].gt(0).all()
