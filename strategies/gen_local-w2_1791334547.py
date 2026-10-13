from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class OrdinalMomentumParams:
    rank_window: int = 63
    persistence_window: int = 18


class GeneratedStrategy(BaseStrategy[OrdinalMomentumParams]):
    strategy_id = "gen_local-w2_1791334547"

    @classmethod
    def params_type(cls) -> type[OrdinalMomentumParams]:
        return OrdinalMomentumParams

    def warmup_bars(self, params: OrdinalMomentumParams) -> int:
        return max(2, int(params.rank_window)) + max(1, int(params.persistence_window))

    def indicators(self, data: pd.DataFrame, params: OrdinalMomentumParams) -> pd.DataFrame:
        rank_window = max(2, int(params.rank_window))
        persistence_window = max(1, int(params.persistence_window))
        out = pd.DataFrame(index=data.index)
        out["daily_return"] = data["close"].pct_change(fill_method=None).replace([np.inf, -np.inf], np.nan)
        out["return_rank"] = out["daily_return"].rolling(rank_window, min_periods=rank_window).rank(method="average")
        out["return_percentile"] = (out["return_rank"] - 1.0) / (rank_window - 1.0)
        out["ordinal_pressure"] = (out["return_percentile"] - 0.5).rolling(persistence_window, min_periods=persistence_window).mean()
        return out

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: OrdinalMomentumParams) -> SignalFrame:
        pressure = indicators["ordinal_pressure"]
        # Equal thresholds: positive pressure enters; negative pressure exits.
        # Exact ties and unavailable observations preserve the existing position.
        target = pd.Series(
            np.where(pressure.gt(0.0), 1.0, np.where(pressure.lt(0.0), 0.0, np.nan)),
            index=data.index,
        )
        df = pd.DataFrame(index=data.index)
        df["signal"] = target.ffill().fillna(0).astype(int)
        df["size"] = 1.0
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


if __name__ == "__main__":
    index = pd.date_range("2020-01-01", periods=6)
    data = pd.DataFrame(index=index)
    indicators = pd.DataFrame({"ordinal_pressure": [0.2, 0.0, -0.2, 0.0, 0.2, -0.2]}, index=index)
    strategy = GeneratedStrategy()
    result = strategy.generate_signals(data, indicators, None, OrdinalMomentumParams())
    assert result.data["signal"].tolist() == [0, 1, 1, 0, 0, 1]
    assert result.data["size"].gt(0).all()
