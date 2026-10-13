from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class RangeFloorParams:
    box_bars: int = 18
    rank_bars: int = 126
    compression_percentile: float = 0.35

    def __post_init__(self) -> None:
        if self.box_bars < 2 or self.rank_bars < 2:
            raise ValueError("Lookbacks must be at least two bars.")
        if not 0.0 < self.compression_percentile <= 1.0:
            raise ValueError("Compression percentile must be in (0, 1].")


class GeneratedStrategy(BaseStrategy[RangeFloorParams]):
    strategy_id = "gen_local_1790867713"

    @classmethod
    def params_type(cls) -> type[RangeFloorParams]:
        return RangeFloorParams

    def warmup_bars(self, params: RangeFloorParams) -> int:
        return int(max(params.box_bars + params.rank_bars - 1,
                       2 * params.box_bars) + 2)

    def indicators(self, data: pd.DataFrame, params: RangeFloorParams) -> pd.DataFrame:
        out = pd.DataFrame(index=data.index)
        out["daily_range"] = (data["high"] - data["low"]).clip(lower=0.0)
        out["floor"] = data["low"].rolling(params.box_bars).min()
        out["ceiling"] = data["high"].rolling(params.box_bars).max()
        out["range_sum"] = out["daily_range"].rolling(params.box_bars).sum()
        out["packing"] = (out["ceiling"] - out["floor"]) / out["range_sum"].where(out["range_sum"] > 0.0)
        out["compression_rank"] = out["packing"].rolling(params.rank_bars).rank(pct=True)
        out["floor_change"] = out["floor"].diff(params.box_bars)
        out["ceiling_change"] = out["ceiling"].diff(params.box_bars)
        return out.replace([np.inf, -np.inf], np.nan)

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: RangeFloorParams,
    ) -> SignalFrame:
        eligible = (
            indicators.notna().all(axis=1)
            & (indicators["compression_rank"] <= params.compression_percentile)
            & (indicators["floor_change"] > 0.0)
            & (indicators["floor_change"] > indicators["ceiling_change"].clip(lower=0.0))
        ).fillna(False)
        # Two closes enter; the first false qualification exits without a timer.
        confirmed = eligible & eligible.shift(1, fill_value=False)
        df = pd.DataFrame(index=data.index)
        df["signal"] = confirmed.astype(int)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = 0.95
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check() -> None:
    strategy = GeneratedStrategy()
    params = RangeFloorParams()
    index = pd.date_range("2020-01-01", periods=7)
    data = pd.DataFrame(100.0, index=index, columns=["open", "high", "low", "close", "volume"])
    indicators = pd.DataFrame(
        {"compression_rank": [0.2, 0.2, 0.8, 0.2, 0.2, 0.2, 0.8],
         "floor_change": 1.0, "ceiling_change": 0.0}, index=index
    )
    result = strategy.generate_signals(data, indicators, None, params).data
    assert result["signal"].tolist() == [0, 0, 1, 0, 0, 1, 1]
    flat = strategy.indicators(data, params)
    assert strategy.generate_signals(data, flat, None, params).data["signal"].eq(0).all()
    assert result["size"].gt(0.0).all()


if __name__ == "__main__":
    _self_check()
