from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class VisibilityParams:
    graph_window: int = 5
    target_vol: float = 0.15


class GeneratedStrategy(BaseStrategy[VisibilityParams]):
    strategy_id = "gen_local_1790899431"

    @classmethod
    def params_type(cls):
        return VisibilityParams

    def warmup_bars(self, params):
        return int(max(20, params.graph_window) + 1)

    def indicators(self, data, params):
        if params.graph_window < 3 or not np.isfinite(params.target_vol) or params.target_vol <= 0:
            raise ValueError("graph_window must be >= 3 and target_vol must be positive and finite")
        close = data["close"].astype(float).where(data["close"] > 0)
        ret = close.pct_change(fill_method=None).replace([np.inf, -np.inf], np.nan)
        upper = pd.Series(0.0, index=data.index)
        lower = pd.Series(0.0, index=data.index)
        for lag in range(1, params.graph_window):
            past = ret.shift(lag)
            if lag == 1:
                visible = ret.notna() & past.notna()
                upper += visible.astype(float)
                lower += visible.astype(float)
            else:
                ceiling = ret.shift(1).rolling(lag - 1, min_periods=lag - 1).max()
                floor = ret.shift(1).rolling(lag - 1, min_periods=lag - 1).min()
                upper += (np.minimum(ret, past) > ceiling).astype(float)
                lower += (np.maximum(ret, past) < floor).astype(float)
        complete = ret.rolling(params.graph_window, min_periods=params.graph_window).count().eq(params.graph_window)
        upper = upper.where(complete)
        lower = lower.where(complete)
        topology = (lower - upper).where(complete)
        volatility = ret.rolling(20, min_periods=20).std(ddof=0) * np.sqrt(252.0)
        exposure = (params.target_vol / volatility.clip(lower=0.0001)).clip(lower=0.01, upper=0.95)
        return pd.DataFrame({
            "return": ret,
            "upper_degree": upper,
            "lower_degree": lower,
            "topology": topology,
            "volatility": volatility,
            "exposure": exposure,
        }, index=data.index)

    def generate_signals(self, data, indicators, ctx, params):
        ready = indicators["topology"].notna() & indicators["volatility"].notna()
        candidate = np.sign(indicators["topology"].where(ready, 0.0)).astype(int).to_numpy()
        decisions = np.zeros(len(data), dtype=int)
        position = 0
        for i, side in enumerate(candidate):
            if position == 0:
                position = int(side)
            elif side == -position:
                position = 0
            decisions[i] = position
        df = pd.DataFrame({"signal": decisions}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = indicators["exposure"].shift(1).fillna(0.01).astype(float)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check():
    strategy = GeneratedStrategy()
    params = VisibilityParams()
    index = pd.date_range("2020-01-01", periods=64, freq="D")
    close = 100.0 * np.exp(np.cumsum(0.008 * np.sin(np.arange(64) * 1.7)))
    data = pd.DataFrame({"open": close, "high": close * 1.01, "low": close * 0.99, "close": close, "volume": 1000000.0}, index=index)
    indicators = strategy.indicators(data, params)
    prefix = strategy.indicators(data.iloc[:40], params)
    assert np.allclose(indicators.iloc[:40].to_numpy(), prefix.to_numpy(), equal_nan=True)
    ctx = StrategyContext(symbol="SPY", timeframe="1d", warmup_bars=strategy.warmup_bars(params))
    result = strategy.generate_signals(data, indicators, ctx, params).data
    assert result["signal"].isin([-1, 0, 1]).all()
    assert np.isfinite(result["size"]).all() and result["size"].gt(0).all()
    assert result["signal"].iloc[:21].eq(0).all()
    scenario = pd.DataFrame({"topology": [1.0, 0.0, 1.0, -1.0, -1.0, 0.0, 1.0, 1.0], "volatility": 0.2, "exposure": 0.5}, index=index[:8])
    actual = strategy.generate_signals(data.iloc[:8], scenario, ctx, params).data
    assert actual["signal"].tolist() == [0, 1, 1, 1, 0, -1, -1, 0]
    assert actual["size"].tolist() == [0.01] + [0.5] * 7


if __name__ == "__main__":
    _self_check()
