from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class SignatureSeasonParams:
    path_bars: int = 3
    area_threshold: float = 0.08


class GeneratedStrategy(BaseStrategy[SignatureSeasonParams]):
    strategy_id = "gen_local_1791340589"

    @classmethod
    def params_type(cls) -> type[SignatureSeasonParams]:
        return SignatureSeasonParams

    def warmup_bars(self, params: SignatureSeasonParams) -> int:
        return max(2, int(params.path_bars)) + 2

    def indicators(self, data: pd.DataFrame, params: SignatureSeasonParams) -> pd.DataFrame:
        window = max(2, int(params.path_bars))
        close = data["close"].astype(float)
        log_price = np.log(close.where(close.gt(0))).replace([np.inf, -np.inf], np.nan)
        increments = log_price.diff()
        displacement = increments.rolling(window, min_periods=window).sum()
        variation = increments.abs().rolling(window, min_periods=window).sum()
        area = pd.Series(0.0, index=data.index)
        # Half the antisymmetric second-level signature of (time, log price).
        # Increment j has weight (2*j - window - 1)/2, oldest first.
        for j in range(1, window + 1):
            area = area + 0.5 * (2 * j - window - 1) * increments.shift(window - j)
        normalized_area = area.div(window * variation.where(variation.gt(0)))
        weekday = pd.Series(data.index.dayofweek, index=data.index)
        return pd.DataFrame({
            "signature_area": normalized_area.replace([np.inf, -np.inf], np.nan),
            "displacement": displacement,
            "season_eligible": weekday.between(1, 3),
        }, index=data.index)

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame,
                         ctx: StrategyContext, params: SignatureSeasonParams) -> SignalFrame:
        condition = (
            indicators["season_eligible"].fillna(False).astype(bool)
            & indicators["signature_area"].gt(params.area_threshold)
            & indicators["displacement"].lt(0)
        ).fillna(False)
        # Two consecutive true closes enter; the first false close exits.
        confirmed = condition & condition.shift(1, fill_value=False)
        df = pd.DataFrame(index=data.index)
        df["signal"] = confirmed.astype(int)
        df["size"] = 0.95
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check() -> None:
    index = pd.bdate_range("2024-01-08", periods=8)
    data = pd.DataFrame({"open": 100.0, "high": 101.0, "low": 98.0,
                         "close": 100.0, "volume": 1000.0}, index=index)
    strategy = GeneratedStrategy()
    params = SignatureSeasonParams(path_bars=2)
    data.loc[index[1]:, "close"] = 99.0
    derived = strategy.indicators(data, params)
    assert np.isclose(derived["signature_area"].iloc[2], 0.25)
    assert pd.isna(derived["signature_area"].iloc[0])
    conditions = pd.DataFrame({
        "signature_area": [0.0, 0.2, 0.2, 0.2, 0.2, 0.0, 0.2, 0.2],
        "displacement": -0.01,
        "season_eligible": index.dayofweek >= 1,
    }, index=index)
    conditions["season_eligible"] &= index.dayofweek <= 3
    result = strategy.generate_signals(data, conditions, None, params).data
    assert result["signal"].tolist() == [0, 0, 0, 1, 1, 0, 0, 0]
    assert result["size"].gt(0).all()
    assert strategy.warmup_bars(params) >= params.path_bars + 1


if __name__ == "__main__":
    _self_check()
