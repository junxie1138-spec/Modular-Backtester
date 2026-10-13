from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class VisibilityParams:
    lookback: int = 7
    degree_advantage: int = 2
    max_position: float = 0.45


class GeneratedStrategy(BaseStrategy[VisibilityParams]):
    strategy_id = "gen_local_1791332574"

    @classmethod
    def params_type(cls) -> type[VisibilityParams]:
        return VisibilityParams

    @staticmethod
    def warmup_bars(params: VisibilityParams) -> int:
        if not 3 <= params.lookback <= 10:
            raise ValueError("lookback must be between 3 and 10")
        if params.degree_advantage < 1:
            raise ValueError("degree_advantage must be positive")
        if not 0.0 <= params.max_position <= 1.0:
            raise ValueError("max_position must be between zero and one")
        return int(params.lookback)

    @staticmethod
    def indicators(data: pd.DataFrame, params: VisibilityParams) -> pd.DataFrame:
        n = GeneratedStrategy.warmup_bars(params)
        close = data["close"].astype(float).replace([np.inf, -np.inf], np.nan)
        close = close.where(close > 0.0)
        price = np.log(close)
        pivot = price.shift(1)
        # Both adjacent nodes connect to the penultimate node.
        upper = pd.Series(2.0, index=data.index)
        lower = upper.copy()
        for lag in range(2, n - 1):
            past = price.shift(lag + 1)
            upper_edge = past.notna() & pivot.notna()
            lower_edge = upper_edge.copy()
            for offset in range(1, lag):
                chord = pivot + (past - pivot) * (offset / lag)
                intermediate = price.shift(offset + 1)
                upper_edge &= intermediate < chord
                lower_edge &= intermediate > chord
            upper += upper_edge.astype(float)
            lower += lower_edge.astype(float)
        valid = price.rolling(n, min_periods=n).count().eq(n)
        floor = close.rolling(n, min_periods=n).min()
        ceiling = close.rolling(n, min_periods=n).max()
        span = (ceiling - floor).replace(0.0, np.nan)
        position = (close - floor) / span
        return pd.DataFrame({
            "upper_degree": upper.where(valid),
            "lower_degree": lower.where(valid),
            "degree_advantage": (lower - upper).where(valid),
            "relative_position": position.where(valid),
            "rebound": (close > close.shift(1)).where(valid, False),
            "ready": valid,
        }, index=data.index)

    @staticmethod
    def generate_signals(
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: VisibilityParams,
    ) -> SignalFrame:
        entry = (
            indicators["ready"].fillna(False)
            & indicators["degree_advantage"].ge(params.degree_advantage)
            & indicators["relative_position"].le(params.max_position)
            & indicators["rebound"].fillna(False)
        )
        df = pd.DataFrame(index=data.index)
        # The position exits exactly when the combined entry condition flips false.
        df["signal"] = entry.fillna(False).astype(int)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = 1.0
        return SignalFrame(data=df, signal_column="signal", size_column="size")


if __name__ == "__main__":
    params = VisibilityParams()
    closes = [10.0, 9.0, 8.0, 7.0, 6.0, 5.0, 5.5, 6.0, 6.5]
    sample = pd.DataFrame({"close": closes}, index=pd.date_range("2020-01-01", periods=len(closes)))
    derived = GeneratedStrategy.indicators(sample, params)
    result = GeneratedStrategy.generate_signals(sample, derived, None, params).data
    assert GeneratedStrategy.warmup_bars(params) <= 10
    assert derived["lower_degree"].iloc[6] > derived["upper_degree"].iloc[6]
    assert result["signal"].iloc[:7].eq(0).all()
    assert result["signal"].iloc[7] == 1 and result["signal"].iloc[8] == 0
    prefix = GeneratedStrategy.indicators(sample.iloc[:7], params)
    pd.testing.assert_frame_equal(derived.iloc[:7], prefix)
    assert result["signal"].isin([0, 1]).all() and result["size"].gt(0).all()
