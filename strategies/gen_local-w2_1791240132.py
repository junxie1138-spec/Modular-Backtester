from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class BalancedTrendParams:
    leg_bars: int = 5
    rank_bars: int = 63
    entry_percentile: float = 0.65
    min_efficiency: float = 0.20

    def __post_init__(self):
        if self.leg_bars < 2 or self.rank_bars < 2:
            raise ValueError("Lookbacks must be at least two bars.")
        if not 0.0 < self.entry_percentile <= 1.0:
            raise ValueError("entry_percentile must be in (0, 1].")
        if not 0.0 <= self.min_efficiency < 1.0:
            raise ValueError("min_efficiency must be in [0, 1).")


class GeneratedStrategy(BaseStrategy[BalancedTrendParams]):
    strategy_id = "gen_local-w2_1791240132"

    @classmethod
    def params_type(cls):
        return BalancedTrendParams

    def warmup_bars(self, params: BalancedTrendParams) -> int:
        return int(2 * params.leg_bars + params.rank_bars + 1)

    def indicators(self, data: pd.DataFrame, params: BalancedTrendParams) -> pd.DataFrame:
        out = pd.DataFrame(index=data.index)
        close = data["close"].where(np.isfinite(data["close"]) & data["close"].gt(0))
        out["log_return"] = np.log(close).diff()
        out["leg_gain"] = out["log_return"].rolling(params.leg_bars, min_periods=params.leg_bars).sum()
        out["leg_path"] = out["log_return"].abs().rolling(params.leg_bars, min_periods=params.leg_bars).sum()
        out["recent_efficiency"] = out["leg_gain"] / out["leg_path"].replace(0.0, np.nan)
        out["prior_efficiency"] = out["recent_efficiency"].shift(params.leg_bars)
        out["weakest_efficiency"] = out[["recent_efficiency", "prior_efficiency"]].min(axis=1, skipna=False)
        out["strength_rank"] = out["weakest_efficiency"].rolling(params.rank_bars, min_periods=params.rank_bars).rank(pct=True)
        return out

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: BalancedTrendParams,
    ) -> SignalFrame:
        eligible = (
            indicators["weakest_efficiency"].gt(params.min_efficiency)
            & indicators["strength_rank"].ge(params.entry_percentile)
        ).fillna(False)
        # Two consecutive eligible closes enter; the first ineligible close exits.
        confirmed = eligible & eligible.shift(1, fill_value=False)
        df = pd.DataFrame(index=data.index)
        df["signal"] = confirmed.astype(int)
        df["size"] = 0.95
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check():
    strategy = object.__new__(GeneratedStrategy)
    params = BalancedTrendParams()
    index = pd.date_range("2020-01-01", periods=8)
    data = pd.DataFrame(index=index)
    indicators = pd.DataFrame(
        {"weakest_efficiency": [0.3, 0.3, 0.3, 0.3, 0.0, 0.3, 0.3, 0.3],
         "strength_rank": [0.8] * 8},
        index=index,
    )
    result = strategy.generate_signals(data, indicators, None, params).data
    assert result["signal"].tolist() == [0, 0, 1, 1, 1, 0, 0, 1]
    assert result["size"].gt(0).all()
    flat = pd.DataFrame({"close": np.full(100, 100.0)}, index=pd.date_range("2020-01-01", periods=100))
    flat_indicators = strategy.indicators(flat, params)
    assert strategy.generate_signals(flat, flat_indicators, None, params).data["signal"].eq(0).all()


if __name__ == "__main__":
    _self_check()
