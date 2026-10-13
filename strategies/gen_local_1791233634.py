from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class CoverageParams:
    carry_floor: float = 0.0005
    min_dip_fraction: float = 0.25


class GeneratedStrategy(BaseStrategy[CoverageParams]):
    strategy_id = "gen_local_1791233634"

    @classmethod
    def params_type(cls) -> type[CoverageParams]:
        return CoverageParams

    @staticmethod
    def warmup_bars(params: CoverageParams) -> int:
        return 4

    @staticmethod
    def indicators(data: pd.DataFrame, params: CoverageParams) -> pd.DataFrame:
        if not np.isfinite(params.carry_floor) or params.carry_floor < 0:
            raise ValueError("carry_floor must be finite and nonnegative")
        if not np.isfinite(params.min_dip_fraction) or not 0 <= params.min_dip_fraction <= 1:
            raise ValueError("min_dip_fraction must be between zero and one")
        returns = data["close"].pct_change(fill_method=None).replace([np.inf, -np.inf], np.nan)
        pair_return = returns.rolling(2, min_periods=2).sum()
        coverage_floor = pair_return.rolling(2, min_periods=2).min()
        minimum_return = returns.rolling(3, min_periods=3).min()
        maximum_return = returns.rolling(3, min_periods=3).max()
        dip_fraction = (-minimum_return.clip(upper=0)).div(maximum_return.where(maximum_return > 0))
        return pd.DataFrame({
            "return": returns,
            "pair_return": pair_return,
            "coverage_floor": coverage_floor,
            "minimum_return": minimum_return,
            "maximum_return": maximum_return,
            "dip_fraction": dip_fraction,
        }, index=data.index)

    @staticmethod
    def generate_signals(data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: CoverageParams) -> SignalFrame:
        ready = indicators.notna().all(axis=1)
        entry = (
            ready
            & indicators["return"].gt(0)
            & indicators["minimum_return"].lt(0)
            & indicators["coverage_floor"].gt(params.carry_floor)
            & indicators["dip_fraction"].ge(params.min_dip_fraction)
        ).fillna(False)
        df = pd.DataFrame(index=data.index)
        # The same condition controls entry and reversal exit; no timed exits.
        # A required down day within three returns limits consecutive exposure to two bars.
        df["signal"] = entry.astype(int)
        df["size"] = 0.95
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


if __name__ == "__main__":
    close = pd.Series([100.0, 102.0, 101.0, 103.0, 104.0, 105.0, 106.0], index=pd.date_range("2020-01-01", periods=7))
    data = pd.DataFrame({"open": close, "high": close, "low": close, "close": close, "volume": 1000.0})
    params = CoverageParams()
    derived = GeneratedStrategy.indicators(data, params)
    result = GeneratedStrategy.generate_signals(data, derived, None, params).data
    assert result["signal"].tolist() == [0, 0, 0, 0, 1, 1, 0]
    assert result["size"].gt(0).all()
    assert GeneratedStrategy.warmup_bars(params) == 4
