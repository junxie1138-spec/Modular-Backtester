from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class DrawdownOccupancyParams:
    peak_window: int = 20
    occupancy_window: int = 5
    quorum: int = 4
    shallow_depth: float = 0.004
    deep_depth: float = 0.03
    holding_bars: int = 4


class GeneratedStrategy(BaseStrategy[DrawdownOccupancyParams]):
    strategy_id = "gen_local_1790998444"

    @classmethod
    def params_type(cls) -> type[DrawdownOccupancyParams]:
        return DrawdownOccupancyParams

    def warmup_bars(self, params: DrawdownOccupancyParams) -> int:
        return int(max(200, params.peak_window + params.occupancy_window))

    def indicators(self, data: pd.DataFrame, params: DrawdownOccupancyParams) -> pd.DataFrame:
        if params.peak_window < 1 or params.occupancy_window < 1:
            raise ValueError("Lookback windows must be positive.")
        if not 1 <= params.quorum <= params.occupancy_window:
            raise ValueError("Quorum must fit inside the occupancy window.")
        if not 0 <= params.shallow_depth < params.deep_depth < 1:
            raise ValueError("Drawdown thresholds must be ordered between zero and one.")
        if not 3 <= params.holding_bars <= 5:
            raise ValueError("Holding period must be three to five bars.")

        close = data["close"].astype(float).replace([np.inf, -np.inf], np.nan)
        close = close.where(close > 0)
        peak = close.rolling(params.peak_window, min_periods=params.peak_window).max()
        depth = (1.0 - close / peak).clip(lower=0.0)
        ma200 = close.rolling(200, min_periods=200).mean()
        shallow = depth.le(params.shallow_depth).astype(float).where(depth.notna())
        deep = depth.ge(params.deep_depth).astype(float).where(depth.notna())
        shallow_count = shallow.rolling(params.occupancy_window, min_periods=params.occupancy_window).sum()
        deep_count = deep.rolling(params.occupancy_window, min_periods=params.occupancy_window).sum()

        return pd.DataFrame({
            "rolling_peak": peak,
            "drawdown_depth": depth,
            "ma200": ma200,
            "shallow_occupancy": shallow_count,
            "deep_occupancy": deep_count,
            "previous_shallow_occupancy": shallow_count.shift(1),
            "previous_deep_occupancy": deep_count.shift(1),
        }, index=data.index)

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: DrawdownOccupancyParams) -> SignalFrame:
        ready = indicators.notna().all(axis=1) & data["close"].notna()
        long_entry = (
            ready
            & data["close"].gt(indicators["ma200"])
            & indicators["shallow_occupancy"].ge(params.quorum)
            & indicators["previous_shallow_occupancy"].lt(params.quorum)
        )
        short_entry = (
            ready
            & data["close"].lt(indicators["ma200"])
            & indicators["deep_occupancy"].ge(params.quorum)
            & indicators["previous_deep_occupancy"].lt(params.quorum)
        )
        entries = np.select([long_entry, short_entry], [1, -1], default=0)
        signal = np.zeros(len(data), dtype=np.int64)
        position = 0
        exit_bar = -1
        for i, entry in enumerate(entries):
            # An expiry always produces a flat bar; occupied positions ignore new entries.
            if position != 0 and i >= exit_bar:
                position = 0
            elif position == 0 and entry != 0:
                position = int(entry)
                exit_bar = i + params.holding_bars
            signal[i] = position

        df = pd.DataFrame({"signal": signal, "size": 0.95}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")
