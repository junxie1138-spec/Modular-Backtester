from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class Params:
    flow_bars: int = 4
    rank_window: int = 40
    tail: float = 0.30

    def __post_init__(self):
        if self.flow_bars < 1 or self.rank_window < 2 or not 0.0 < self.tail < 0.5:
            raise ValueError("Require flow_bars >= 1, rank_window >= 2, and 0 < tail < 0.5")


class GeneratedStrategy(BaseStrategy[Params]):
    strategy_id = "gen_local_1791231523"

    @classmethod
    def params_type(cls) -> type[Params]:
        return Params

    def warmup_bars(self, params: Params) -> int:
        return int(params.flow_bars + params.rank_window)

    def indicators(self, data: pd.DataFrame, params: Params) -> pd.DataFrame:
        prices = data[["open", "close"]].where(data[["open", "close"]] > 0)
        intraday = np.log(prices["close"] / prices["open"]).replace([np.inf, -np.inf], np.nan)
        overnight = np.log(prices["open"] / prices["close"].shift(1)).replace([np.inf, -np.inf], np.nan)
        out = pd.DataFrame(index=data.index)
        out["intraday_return"] = intraday
        out["overnight_return"] = overnight
        out["intraday_flow"] = intraday.rolling(params.flow_bars, min_periods=params.flow_bars).sum()
        out["overnight_flow"] = overnight.rolling(params.flow_bars, min_periods=params.flow_bars).sum()
        out["intraday_rank"] = out["intraday_flow"].rolling(
            params.rank_window, min_periods=params.rank_window
        ).rank(method="average", pct=True)
        return out

    def generate_signals(
        self, data: pd.DataFrame, indicators: pd.DataFrame,
        ctx: StrategyContext, params: Params
    ) -> SignalFrame:
        rank = indicators["intraday_rank"]
        night = indicators["overnight_flow"]
        ready = np.isfinite(rank) & np.isfinite(night)
        long_entry = ready & (rank <= params.tail) & (night > 0.0)
        short_entry = ready & (rank >= 1.0 - params.tail) & (night < 0.0)
        entry = np.select([long_entry, short_entry], [1, -1], default=0)
        signals = np.zeros(len(data), dtype=int)
        position = 0
        for i, candidate in enumerate(entry):
            if position == 0:
                position = int(candidate)
            elif candidate == -position:
                position = 0
            signals[i] = position
        df = pd.DataFrame({"signal": signals, "size": 1.0}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


if __name__ == "__main__":
    index = pd.date_range("2020-01-01", periods=7)
    data = pd.DataFrame(index=index)
    derived = pd.DataFrame({
        "intraday_rank": [np.nan, 0.1, 0.5, 0.9, 0.9, 0.5, 0.1],
        "overnight_flow": [np.nan, 0.01, 0.0, -0.01, -0.01, 0.0, 0.01],
    }, index=index)
    result = GeneratedStrategy().generate_signals(data, derived, None, Params()).data
    assert result["signal"].tolist() == [0, 0, 1, 1, 0, -1, -1]
    assert (result["size"] > 0).all()
