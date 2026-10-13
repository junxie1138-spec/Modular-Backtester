from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class DiscoveryParams:
    horizon: int = 18
    rank_window: int = 126
    entry_percentile: float = 0.7

    def __post_init__(self):
        if self.horizon < 2 or self.rank_window < 2:
            raise ValueError("Lookbacks must be at least two bars.")
        if not 0.5 < self.entry_percentile <= 1.0:
            raise ValueError("entry_percentile must be in (0.5, 1].")


class GeneratedStrategy(BaseStrategy[DiscoveryParams]):
    strategy_id = "gen_local_1790894083"

    @classmethod
    def params_type(cls):
        return DiscoveryParams

    def warmup_bars(self, params: DiscoveryParams) -> int:
        return int(params.horizon + params.rank_window + 1)

    def indicators(self, data: pd.DataFrame, params: DiscoveryParams) -> pd.DataFrame:
        out = pd.DataFrame(index=data.index)
        out["prior_high"] = data["high"].shift(1)
        out["prior_low"] = data["low"].shift(1)
        out["upside_acceptance"] = (data["close"] - out["prior_high"]).clip(lower=0.0)
        out["downside_acceptance"] = (out["prior_low"] - data["close"]).clip(lower=0.0)
        out["close_travel"] = data["close"].diff().abs()
        out["net_discovery"] = (
            out["upside_acceptance"] - out["downside_acceptance"]
        ).rolling(params.horizon, min_periods=params.horizon).sum()
        out["total_travel"] = out["close_travel"].rolling(
            params.horizon, min_periods=params.horizon
        ).sum()
        out["discovery_efficiency"] = (
            out["net_discovery"] / out["total_travel"].replace(0.0, np.nan)
        ).replace([np.inf, -np.inf], np.nan)
        out["discovery_percentile"] = out["discovery_efficiency"].rolling(
            params.rank_window, min_periods=params.rank_window
        ).rank(method="average", pct=True)
        return out

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: DiscoveryParams,
    ) -> SignalFrame:
        qualifies = (
            indicators["discovery_percentile"].ge(params.entry_percentile)
            & indicators["discovery_efficiency"].gt(0.0)
        ).fillna(False)
        df = pd.DataFrame(index=data.index)
        # Hold while the entry condition remains true; its reversal is the only exit.
        df["signal"] = qualifies.astype(int)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = 0.95
        return SignalFrame(data=df, signal_column="signal", size_column="size")


if __name__ == "__main__":
    strategy = GeneratedStrategy()
    params = DiscoveryParams()
    t = np.arange(strategy.warmup_bars(params) + 100, dtype=float)
    close = 100.0 + 0.06 * t + 2.0 * np.sin(t / 9.0)
    data = pd.DataFrame(
        {"open": close, "high": close + 0.05, "low": close - 0.05,
         "close": close, "volume": 1000000.0},
        index=pd.date_range("2020-01-01", periods=len(t), freq="B"),
    )
    ctx = StrategyContext("SPY", "1d", strategy.warmup_bars(params))
    ind = strategy.indicators(data, params)
    signals = strategy.generate_signals(data, ind, ctx, params).data
    expected = (
        ind["discovery_percentile"].ge(params.entry_percentile)
        & ind["discovery_efficiency"].gt(0.0)
    ).astype(int).shift(1).fillna(0).astype(int)
    assert signals["signal"].equals(expected)
    assert set(signals["signal"]) == {0, 1}
    assert (signals["size"] > 0).all()
    prefix = data.iloc[:-25]
    assert strategy.indicators(prefix, params).equals(ind.iloc[:-25])
