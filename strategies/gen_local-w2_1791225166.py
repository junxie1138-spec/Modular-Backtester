from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class StrategyParams:
    vol_fast: int = 5
    vol_slow: int = 15
    target_vol: float = 0.012


class GeneratedStrategy(BaseStrategy[StrategyParams]):
    strategy_id = "gen_local-w2_1791225166"

    @classmethod
    def params_type(cls):
        return StrategyParams

    def warmup_bars(self, params: StrategyParams) -> int:
        return max(params.vol_slow + 1, 4)

    def indicators(self, data: pd.DataFrame, params: StrategyParams) -> pd.DataFrame:
        out = pd.DataFrame(index=data.index)
        x = np.log(data["close"]).diff()
        y = np.log(data["high"] / data["low"]).diff()
        out["area"] = 0.5 * (
            x.shift(2) * y.shift(1) - y.shift(2) * x.shift(1)
            + x.shift(2) * y - y.shift(2) * x
            + x.shift(1) * y - y.shift(1) * x
        )
        out["fast_vol"] = x.rolling(params.vol_fast).std(ddof=0)
        out["slow_vol"] = x.rolling(params.vol_slow).std(ddof=0)
        return out.replace([np.inf, -np.inf], np.nan)

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: StrategyParams,
    ) -> SignalFrame:
        df = pd.DataFrame(index=data.index)
        # A false entry condition reverses the position to flat.
        df["signal"] = (
            indicators["area"].lt(0)
            & indicators["fast_vol"].lt(indicators["slow_vol"])
        ).astype(int)
        df["size"] = (
            params.target_vol / indicators["slow_vol"].replace(0, np.nan)
        ).clip(lower=0.1, upper=1.0).fillna(0.1).astype(float)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


if __name__ == "__main__":
    index = pd.date_range("2020-01-01", periods=40)
    close = pd.Series(100 + np.arange(40) * 0.1 + np.sin(np.arange(40)), index=index)
    bars = pd.DataFrame({"open": close, "high": close + 1, "low": close - 1, "close": close, "volume": 1000}, index=index)
    strategy = GeneratedStrategy()
    params = StrategyParams()
    derived = strategy.indicators(bars, params)
    result = strategy.generate_signals(bars, derived, StrategyContext("SPY", "1d", strategy.warmup_bars(params)), params).data
    expected = (derived["area"].lt(0) & derived["fast_vol"].lt(derived["slow_vol"])).astype(int).shift(1).fillna(0).astype(int)
    assert result["signal"].equals(expected) and result["size"].gt(0).all()
