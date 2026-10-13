from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class RankCapacityParams:
    rank_window: int = 40
    rank_gap: float = 0.25

    def __post_init__(self):
        if self.rank_window < 2 or not 0.0 < self.rank_gap < 1.0:
            raise ValueError("rank_window must be >= 2 and rank_gap between 0 and 1")


class GeneratedStrategy(BaseStrategy[RankCapacityParams]):
    strategy_id = "gen_local-w2_1791318163"

    @classmethod
    def params_type(cls):
        return RankCapacityParams

    def warmup_bars(self, params: RankCapacityParams) -> int:
        return int(params.rank_window + 1)

    def indicators(self, data: pd.DataFrame, params: RankCapacityParams) -> pd.DataFrame:
        out = pd.DataFrame(index=data.index)
        out["body"] = data["close"] / data["open"].where(data["open"] > 0) - 1.0
        out["magnitude"] = out["body"].abs()
        out["body_rank"] = out["magnitude"].rolling(params.rank_window).rank(pct=True)
        out["volume_rank"] = data["volume"].where(data["volume"] > 0).rolling(params.rank_window).rank(pct=True)
        out["rank_excess"] = out["body_rank"] - out["volume_rank"]
        spread = data["high"] - data["low"]
        out["close_location"] = (data["close"] - data["low"]) / spread.where(spread > 0)
        out["close_change"] = data["close"].diff()
        return out.replace([np.inf, -np.inf], np.nan)

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame,
                         ctx: StrategyContext, params: RankCapacityParams) -> SignalFrame:
        ready = indicators.notna().all(axis=1)
        pressure = ready & (indicators["body_rank"] >= 0.8) & (indicators["rank_excess"] >= params.rank_gap)
        buy = pressure & (indicators["body"] > 0) & (indicators["close_change"] > 0) & (indicators["close_location"] >= 0.8)
        sell = pressure & (indicators["body"] < 0) & (indicators["close_change"] < 0) & (indicators["close_location"] <= 0.2)
        entries = np.select([buy, sell], [1, -1], default=0)
        signals = np.zeros(len(data), dtype=int)
        position = 0
        entry_bar = -1
        for i, candidate in enumerate(entries):
            if position:
                if i - entry_bar == 2:
                    position = 0
                    continue
            elif candidate:
                position = int(candidate)
                entry_bar = i
            signals[i] = position
        df = pd.DataFrame({"signal": signals, "size": 1.0}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


if __name__ == "__main__":
    index = pd.date_range("2020-01-01", periods=7)
    data = pd.DataFrame(index=index)
    direction = pd.Series([1, -1, -1, -1, -1, 1, 1], index=index)
    derived = pd.DataFrame({
        "body": direction * 0.01,
        "body_rank": 1.0,
        "rank_excess": 0.5,
        "close_change": direction,
        "close_location": np.where(direction > 0, 0.9, 0.1),
    }, index=index)
    derived.iloc[1] = np.nan
    result = GeneratedStrategy.generate_signals(None, data, derived, None, RankCapacityParams()).data
    assert result["signal"].tolist() == [0, 1, 1, 0, -1, -1, 0]
    assert result["size"].gt(0).all()
