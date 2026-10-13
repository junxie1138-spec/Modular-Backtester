from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class WickRankParams:
    return_window: int = 8
    rank_window: int = 32
    wick_window: int = 3
    tail_fraction: float = 0.20
    wick_threshold: float = 0.12


class GeneratedStrategy(BaseStrategy[WickRankParams]):
    strategy_id = "gen_local-w2_1791235847"

    @classmethod
    def params_type(cls) -> type[WickRankParams]:
        return WickRankParams

    def warmup_bars(self, params: WickRankParams) -> int:
        return int(max(params.return_window + params.rank_window, params.wick_window))

    def indicators(self, data: pd.DataFrame, params: WickRankParams) -> pd.DataFrame:
        if min(params.return_window, params.rank_window, params.wick_window) < 1:
            raise ValueError("Lookbacks must be positive.")
        if not 0 < params.tail_fraction < 0.5 or not 0 < params.wick_threshold < 1:
            raise ValueError("Invalid signal thresholds.")
        out = pd.DataFrame(index=data.index)
        out["multiweek_return"] = (
            data["close"] / data["close"].shift(params.return_window).replace(0, np.nan) - 1
        ).replace([np.inf, -np.inf], np.nan)
        out["return_rank"] = out["multiweek_return"].rolling(
            params.rank_window, min_periods=params.rank_window
        ).rank(method="average", pct=True)
        out["lower_wick"] = (
            data[["open", "close"]].min(axis=1, skipna=False) - data["low"]
        ).clip(lower=0)
        out["upper_wick"] = (
            data["high"] - data[["open", "close"]].max(axis=1, skipna=False)
        ).clip(lower=0)
        out["range"] = (data["high"] - data["low"]).clip(lower=0)
        out["wick_difference"] = out["lower_wick"] - out["upper_wick"]
        out["wick_sum"] = out["wick_difference"].rolling(
            params.wick_window, min_periods=params.wick_window
        ).sum()
        out["range_sum"] = out["range"].rolling(
            params.wick_window, min_periods=params.wick_window
        ).sum()
        out["wick_balance"] = (
            out["wick_sum"] / out["range_sum"].replace(0, np.nan)
        ).replace([np.inf, -np.inf], np.nan)
        return out

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: WickRankParams,
    ) -> SignalFrame:
        rank = indicators["return_rank"]
        wick = indicators["wick_balance"]
        valid = np.isfinite(rank) & np.isfinite(wick)
        buy = valid & (rank <= params.tail_fraction) & (wick >= params.wick_threshold)
        sell = valid & (rank >= 1 - params.tail_fraction) & (wick <= -params.wick_threshold)
        setup = np.select([buy, sell], [1, -1], default=0)
        positions = np.zeros(len(data), dtype=int)
        position = 0
        for i, entry in enumerate(setup):
            if position == 0:
                position = int(entry)
            elif entry == -position:
                # Opposite agreement exits; a new entry needs a subsequent bar.
                position = 0
            positions[i] = position
        df = pd.DataFrame({"signal": positions, "size": 1.0}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


if __name__ == "__main__":
    strategy = object.__new__(GeneratedStrategy)
    params = WickRankParams()
    index = pd.date_range("2020-01-01", periods=9)
    data = pd.DataFrame(index=index)
    derived = pd.DataFrame({
        "return_rank": [0.1, np.nan, 0.1, 0.9, 0.9, 0.5, 0.1, 0.1, 0.9],
        "wick_balance": [0.3, np.nan, -0.3, -0.3, -0.3, 0.0, 0.3, 0.3, 0.3],
    }, index=index)
    result = strategy.generate_signals(data, derived, None, params).data
    assert result["signal"].tolist() == [0, 1, 1, 1, 0, -1, -1, 0, 1]
    assert (result["size"] > 0).all()
    assert strategy.warmup_bars(params) >= params.return_window + params.rank_window
