from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class DrawdownParams:
    anchor_window: int = 20
    occupancy_window: int = 10
    rank_window: int = 60
    tail_percentile: float = 0.75
    hold_bars: int = 4


class GeneratedStrategy(BaseStrategy[DrawdownParams]):
    strategy_id = "gen_local_1791265943"

    @classmethod
    def params_type(cls) -> type[DrawdownParams]:
        return DrawdownParams

    def warmup_bars(self, params: DrawdownParams) -> int:
        return int(params.anchor_window + params.occupancy_window + params.rank_window + 2)

    def indicators(self, data: pd.DataFrame, params: DrawdownParams) -> pd.DataFrame:
        if min(params.anchor_window, params.occupancy_window, params.rank_window) < 2:
            raise ValueError("Lookback windows must be at least two bars")
        if not 0.5 < params.tail_percentile < 1.0:
            raise ValueError("tail_percentile must be between 0.5 and 1")
        if not 3 <= params.hold_bars <= 5:
            raise ValueError("hold_bars must be between three and five")
        out = pd.DataFrame(index=data.index)
        close = data["close"].replace([np.inf, -np.inf], np.nan).where(data["close"] > 0)
        out["anchor"] = close.rolling(params.anchor_window, min_periods=params.anchor_window).max()
        out["drawdown"] = close / out["anchor"] - 1.0
        out["return"] = close.pct_change(fill_method=None)
        out["underwater"] = (out["drawdown"] < 0).astype(float).where(out["drawdown"].notna())
        out["occupancy"] = out["underwater"].rolling(params.occupancy_window, min_periods=params.occupancy_window).mean()
        out["occupancy_rank"] = out["occupancy"].rolling(params.rank_window, min_periods=params.rank_window).rank(pct=True)
        out["depth_rank"] = out["drawdown"].rolling(params.rank_window, min_periods=params.rank_window).rank(pct=True)
        out["ready"] = out[["drawdown", "return", "occupancy_rank", "depth_rank"]].notna().all(axis=1)
        q = params.tail_percentile
        out["long_setup"] = out["ready"] & (out["drawdown"] < 0) & (out["occupancy_rank"] >= q) & (out["depth_rank"] >= q) & (out["return"] > 0)
        out["short_setup"] = out["ready"] & (out["drawdown"] < 0) & (out["occupancy_rank"] <= 1.0 - q) & (out["depth_rank"] <= 1.0 - q) & (out["return"] < 0)
        out["setup"] = np.select([out["long_setup"], out["short_setup"]], [1, -1], default=0).astype(int)
        return out

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: DrawdownParams) -> SignalFrame:
        if not 3 <= params.hold_bars <= 5:
            raise ValueError("hold_bars must be between three and five")
        setups = indicators["setup"].fillna(0).to_numpy(dtype=int)
        signals = np.zeros(len(data), dtype=int)
        position = 0
        exit_bar = -1
        for i in range(len(data)):
            if position:
                if i == exit_bar:
                    position = 0
            elif i > 0 and setups[i] != 0 and setups[i] == setups[i - 1]:
                position = int(setups[i])
                exit_bar = i + params.hold_bars
            signals[i] = position
        df = pd.DataFrame({"signal": signals, "size": 1.0}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


if __name__ == "__main__":
    index = pd.date_range("2020-01-01", periods=13)
    data = pd.DataFrame({"open": 100.0, "high": 101.0, "low": 99.0, "close": 100.0, "volume": 1000.0}, index=index)
    indicators = pd.DataFrame({"setup": [1, 1, -1, -1, 1, 1, -1, -1, -1, -1, -1, -1, 0]}, index=index)
    strategy = GeneratedStrategy.__new__(GeneratedStrategy)
    result = strategy.generate_signals(data, indicators, None, DrawdownParams())
    assert result.data["signal"].tolist() == [0, 0, 1, 1, 1, 1, 0, 0, -1, -1, -1, -1, 0]
    assert (result.data["size"] > 0).all()
    flat = strategy.indicators(data, DrawdownParams())
    assert flat["setup"].eq(0).all()
