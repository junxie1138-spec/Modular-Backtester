from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class GapSeasonParams:
    scale_window: int = 40
    surplus_threshold: float = 0.25


class GeneratedStrategy(BaseStrategy[GapSeasonParams]):
    strategy_id = "gen_local-w2_1790979384"

    @classmethod
    def params_type(cls) -> type[GapSeasonParams]:
        return GapSeasonParams

    def warmup_bars(self, params: GapSeasonParams) -> int:
        return int(params.scale_window + 1)

    def indicators(self, data: pd.DataFrame, params: GapSeasonParams) -> pd.DataFrame:
        out = pd.DataFrame(index=data.index)
        prior_close = data["close"].shift(1).replace(0.0, np.nan)
        out["gap"] = (data["open"] / prior_close - 1.0).replace([np.inf, -np.inf], np.nan)
        out["gap_scale"] = out["gap"].abs().rolling(params.scale_window, min_periods=params.scale_window).mean()
        weeks = data.index.to_period("W-SUN")
        out["weekly_gap"] = out["gap"].groupby(weeks).cumsum()
        out["weekday"] = data.index.dayofweek
        out["entry_condition"] = (
            out["weekday"].between(0, 3)
            & out["gap"].notna()
            & out["gap_scale"].gt(0.0)
            & out["weekly_gap"].gt(params.surplus_threshold * out["gap_scale"])
        ).fillna(False)
        return out

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: GapSeasonParams) -> SignalFrame:
        condition = indicators["entry_condition"].fillna(False).astype(bool)
        confirmed = condition & condition.shift(1, fill_value=False)
        entry = confirmed & indicators["weekday"].eq(1) & indicators["weekday"].shift(1).eq(0)
        events = pd.Series(np.nan, index=data.index, dtype=float)
        events.loc[~condition] = 0.0
        events.loc[entry] = 1.0
        df = pd.DataFrame(index=data.index)
        df["signal"] = events.ffill().fillna(0).astype(int)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = 0.95
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check() -> None:
    index = pd.bdate_range("2024-01-01", periods=15)
    data = pd.DataFrame({"open": 101.0, "high": 102.0, "low": 99.0, "close": 100.0, "volume": 1000.0}, index=index)
    params = GapSeasonParams(scale_window=2)
    indicators = GeneratedStrategy.indicators(None, data, params)
    result = GeneratedStrategy.generate_signals(None, data, indicators, None, params).data
    assert result["signal"].tolist() == [0] * 7 + [1] * 3 + [0] * 2 + [1] * 3
    broken = indicators.copy()
    broken.loc[index[8], "entry_condition"] = False
    result = GeneratedStrategy.generate_signals(None, data, broken, None, params).data
    assert result["signal"].iloc[8] == 1
    assert result["signal"].iloc[9] == 0
    assert result["signal"].iloc[10] == 0
    assert GeneratedStrategy.warmup_bars(None, params) == 3


if __name__ == "__main__":
    _self_check()
