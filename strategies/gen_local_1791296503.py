from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class RangeOverlapParams:
    lookback: int = 20
    min_overlap: float = 3.0
    edge: float = 0.30
    profit_target: float = 0.05
    hold_bars: int = 18


class GeneratedStrategy(BaseStrategy[RangeOverlapParams]):
    strategy_id = "gen_local_1791296503"

    @classmethod
    def params_type(cls):
        return RangeOverlapParams

    def warmup_bars(self, params: RangeOverlapParams) -> int:
        return params.lookback

    def indicators(self, data: pd.DataFrame, params: RangeOverlapParams) -> pd.DataFrame:
        upper = data["high"].rolling(params.lookback, min_periods=params.lookback).max()
        lower = data["low"].rolling(params.lookback, min_periods=params.lookback).min()
        span = (upper - lower).where(upper > lower)
        daily_range = (data["high"] - data["low"]).clip(lower=0)
        out = pd.DataFrame(index=data.index)
        out["overlap"] = daily_range.rolling(params.lookback, min_periods=params.lookback).sum() / span
        out["edge_position"] = (data["close"] - (upper + lower) / 2) / span
        return out.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: RangeOverlapParams,
    ) -> SignalFrame:
        overlap = indicators["overlap"].to_numpy()
        edge = indicators["edge_position"].to_numpy()
        candidate = np.select(
            [(overlap >= params.min_overlap) & (edge <= -params.edge),
             (overlap >= params.min_overlap) & (edge >= params.edge)],
            [1, -1],
            default=0,
        )
        raw = np.zeros(len(data), dtype=int)
        sizes = np.full(len(data), 0.1, dtype=float)
        closes = data["close"].to_numpy()
        side = 0
        entry_close = 0.0
        entry_bar = 0
        stake = 0.1
        for i in range(len(data)):
            if side:
                gain = side * (closes[i] / entry_close - 1.0)
                if gain >= params.profit_target or i - entry_bar >= params.hold_bars:
                    side = 0
                    stake = 0.1
                    raw[i] = 0
                    sizes[i] = stake
                    continue
            elif i >= params.lookback - 1 and candidate[i]:
                side = int(candidate[i])
                entry_close = float(closes[i])
                entry_bar = i
                stake = float(np.clip(0.15 + 1.6 * abs(edge[i]), 0.15, 0.95))
            raw[i] = side
            sizes[i] = stake
        df = pd.DataFrame({"signal": raw, "size": sizes}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = df["size"].shift(1).fillna(0.1).astype(float)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


if __name__ == "__main__":
    strategy = GeneratedStrategy()
    params = RangeOverlapParams(lookback=2)
    index = pd.date_range("2020-01-01", periods=5)
    data = pd.DataFrame({"open": [100] * 5, "high": [101] * 5,
                         "low": [99] * 5, "close": [100, 100, 100, 106, 106],
                         "volume": [1000] * 5}, index=index)
    indicators = pd.DataFrame({"overlap": [0, 4, 0, 0, 0],
                               "edge_position": [0, -0.4, 0, 0, 0]}, index=index)
    result = strategy.generate_signals(data, indicators, StrategyContext("SPY", "1d", 2), params).data
    assert result["signal"].tolist() == [0, 0, 1, 1, 0]
    assert result["size"].gt(0).all()
