from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class EnvelopeParams:
    displacement_bars: int = 4
    rank_window: int = 16
    entry_percentile: float = 0.65


class GeneratedStrategy(BaseStrategy[EnvelopeParams]):
    strategy_id = "gen_local_1791332481"

    @classmethod
    def params_type(cls) -> type[EnvelopeParams]:
        return EnvelopeParams

    @staticmethod
    def warmup_bars(params: EnvelopeParams) -> int:
        if params.displacement_bars < 1 or params.rank_window < 2:
            raise ValueError("displacement_bars must be positive and rank_window at least two")
        if not 0.5 < params.entry_percentile < 1.0:
            raise ValueError("entry_percentile must lie strictly between 0.5 and 1.0")
        return int(max(200, params.displacement_bars + params.rank_window))

    @staticmethod
    def indicators(data: pd.DataFrame, params: EnvelopeParams) -> pd.DataFrame:
        GeneratedStrategy.warmup_bars(params)
        out = pd.DataFrame(index=data.index)
        out["ma200"] = data["close"].rolling(200, min_periods=200).mean()
        for boundary in ("high", "low"):
            previous = data[boundary].shift(params.displacement_bars).where(lambda s: s > 0)
            change = (data[boundary] / previous - 1.0).replace([np.inf, -np.inf], np.nan)
            out[boundary + "_displacement"] = change
            rank = change.rolling(params.rank_window, min_periods=params.rank_window).rank(method="average")
            out[boundary + "_percentile"] = (rank - 1.0) / (params.rank_window - 1.0)
        return out

    @staticmethod
    def generate_signals(
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: EnvelopeParams,
    ) -> SignalFrame:
        high_rank = indicators["high_percentile"]
        low_rank = indicators["low_percentile"]
        valid = indicators[["ma200", "high_percentile", "low_percentile"]].notna().all(axis=1)
        long_setup = (
            valid
            & (data["close"] > indicators["ma200"])
            & (high_rank >= params.entry_percentile)
            & (low_rank >= params.entry_percentile)
        )
        short_setup = (
            valid
            & (data["close"] < indicators["ma200"])
            & (high_rank <= 1.0 - params.entry_percentile)
            & (low_rank <= 1.0 - params.entry_percentile)
        )
        desired = np.select([long_setup, short_setup], [1, -1], default=0).astype(int)
        positions = np.zeros(len(data), dtype=int)
        position = 0
        for i, direction in enumerate(desired):
            # Exit only when the active entry predicate flips false.
            # A direct opposing setup first produces a flat decision bar.
            if position != 0 and direction != position:
                position = 0
            elif position == 0:
                position = int(direction)
            positions[i] = position
        df = pd.DataFrame({"signal": positions, "size": 1.0}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


if __name__ == "__main__":
    params = EnvelopeParams()
    index = pd.date_range("2020-01-01", periods=8)
    data = pd.DataFrame({"open": 100.0, "high": 101.0, "low": 99.0, "close": 100.0, "volume": 1000.0}, index=index)
    ranks = [np.nan, 0.8, 0.8, 0.5, 0.2, 0.8, 0.8, 0.5]
    derived = pd.DataFrame({"ma200": [np.nan, 90.0, 90.0, 90.0, 110.0, 90.0, 90.0, 90.0], "high_percentile": ranks, "low_percentile": ranks}, index=index)
    result = GeneratedStrategy.generate_signals(data, derived, None, params).data
    assert result["signal"].tolist() == [0, 0, 1, 1, 0, -1, 0, 1]
    assert result["size"].gt(0).all()
    assert GeneratedStrategy.warmup_bars(params) >= params.displacement_bars + params.rank_window
