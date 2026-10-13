from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class ElasticRiskParams:
    horizon: int = 18
    risk_window: int = 36
    hurdle: float = 0.45
    target_vol: float = 0.12
    vol_floor: float = 0.04
    max_size: float = 0.95


class GeneratedStrategy(BaseStrategy[ElasticRiskParams]):
    strategy_id = "gen_local-w2_1791325163"

    @classmethod
    def params_type(cls):
        return ElasticRiskParams

    def warmup_bars(self, params):
        return int(max(params.horizon + 1, params.risk_window + 2))

    def indicators(self, data, params):
        if params.horizon < 2 or params.risk_window < 2:
            raise ValueError("Lookbacks must be at least two.")
        if not all(np.isfinite(x) for x in (params.hurdle, params.target_vol, params.vol_floor, params.max_size)):
            raise ValueError("Parameters must be finite.")
        if params.hurdle < 0 or min(params.target_vol, params.vol_floor, params.max_size) <= 0 or params.max_size > 1:
            raise ValueError("Invalid risk or sizing parameters.")
        out = pd.DataFrame(index=data.index)
        out["return"] = data["close"].pct_change(fill_method=None).replace([np.inf, -np.inf], np.nan)
        r = out["return"]
        lag = r.shift(1)
        w = params.risk_window
        out["drift"] = r.rolling(params.horizon, min_periods=params.horizon).sum()
        out["variance"] = r.rolling(w, min_periods=w).var(ddof=0).clip(lower=0.0)
        out["lag_covariance"] = (r * lag).rolling(w, min_periods=w).mean() - r.rolling(w, min_periods=w).mean() * lag.rolling(w, min_periods=w).mean()
        # ponytail: lag-one covariance only; higher lags if residual dependence matters.
        out["effective_variance"] = (out["variance"] + 2.0 * (1.0 - 1.0 / params.horizon) * out["lag_covariance"]).clip(lower=0.25 * out["variance"])
        out["annual_vol"] = np.sqrt(252.0 * out["effective_variance"]).clip(lower=params.vol_floor)
        horizon_risk = np.sqrt(params.horizon * out["effective_variance"]).clip(lower=params.vol_floor * np.sqrt(params.horizon / 252.0))
        out["score"] = (out["drift"] / horizon_risk - params.hurdle).replace([np.inf, -np.inf], np.nan)
        return out

    def generate_signals(self, data, indicators, ctx, params):
        score = indicators["score"]
        df = pd.DataFrame(index=data.index)
        # The same score controls entry and reversal exit; there is no time exit.
        df["signal"] = score.gt(0).astype(int)
        strength = np.tanh(score.clip(lower=0).fillna(0))
        allocation = (params.target_vol / indicators["annual_vol"] * strength).clip(lower=0.000001, upper=params.max_size)
        df["size"] = allocation.shift(1).fillna(0.000001).astype(float)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


if __name__ == "__main__":
    p = ElasticRiskParams()
    strategy = GeneratedStrategy()
    index = pd.date_range("2020-01-01", periods=180)
    returns = np.r_[np.full(90, 0.003), np.full(90, -0.003)]
    close = pd.Series(100.0 * np.cumprod(1.0 + returns), index=index)
    data = pd.DataFrame({"open": close, "high": close, "low": close, "close": close, "volume": 1000.0})
    ctx = StrategyContext("SPY", "1d", strategy.warmup_bars(p))
    ind = strategy.indicators(data, p)
    result = strategy.generate_signals(data, ind, ctx, p).data
    expected = ind["score"].gt(0).astype(int).shift(1).fillna(0).astype(int)
    assert result["signal"].equals(expected)
    assert result["signal"].iloc[60] == 1 and result["signal"].iloc[-1] == 0
    assert result["signal"].iloc[:strategy.warmup_bars(p)].eq(0).all()
    assert np.isfinite(result["size"]).all() and result["size"].gt(0).all()
    assert result["size"].le(p.max_size).all()
    prefix = data.iloc[:110]
    prefix_result = strategy.generate_signals(prefix, strategy.indicators(prefix, p), ctx, p).data
    assert prefix_result.equals(result.iloc[:110])
