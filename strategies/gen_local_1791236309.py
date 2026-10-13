from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class TopologyParams:
    graph_window: int = 10
    smooth_window: int = 5
    percentile_window: int = 126
    entry_percentile: float = 0.80
    vol_window: int = 20
    target_vol: float = 0.12


class GeneratedStrategy(BaseStrategy[TopologyParams]):
    strategy_id = "gen_local_1791236309"

    @classmethod
    def params_type(cls) -> type[TopologyParams]:
        return TopologyParams

    def warmup_bars(self, params: TopologyParams) -> int:
        return int(max(params.graph_window + params.smooth_window + params.percentile_window,
                       params.vol_window + 1))

    def indicators(self, data: pd.DataFrame, params: TopologyParams) -> pd.DataFrame:
        if (params.graph_window < 3 or params.smooth_window < 1
                or params.percentile_window < 5 or params.vol_window < 2
                or not 0.5 < params.entry_percentile < 1.0
                or not np.isfinite(params.target_vol) or params.target_vol <= 0):
            raise ValueError("Invalid topology or volatility parameters")
        out = pd.DataFrame(index=data.index)
        out["log_close"] = np.log(data["close"].where(data["close"] > 0)).replace(
            [np.inf, -np.inf], np.nan)
        price = out["log_close"]
        upper = pd.Series(-np.inf, index=data.index)
        lower = pd.Series(np.inf, index=data.index)
        up_degree = pd.Series(0.0, index=data.index)
        down_degree = pd.Series(0.0, index=data.index)
        # Endpoint visibility: an edge exists when its slope exceeds all nearer slopes.
        # Negating price converts upper visibility into lower visibility.
        for lag in range(1, params.graph_window):
            slope = (price.shift(lag) - price) / lag
            up_degree += (slope > upper).astype(float)
            down_degree += (slope < lower).astype(float)
            upper = np.maximum(upper, slope)
            lower = np.minimum(lower, slope)
        valid = price.rolling(params.graph_window).count().eq(params.graph_window)
        out["upper_degree"] = up_degree.where(valid)
        out["lower_degree"] = down_degree.where(valid)
        out["degree_balance"] = (
            (out["upper_degree"] - out["lower_degree"]) / (params.graph_window - 2)
        )
        out["topology"] = out["degree_balance"].rolling(params.smooth_window).mean()
        history = out["topology"].shift(1).rolling(params.percentile_window)
        out["upper_threshold"] = history.quantile(params.entry_percentile)
        out["lower_threshold"] = history.quantile(1.0 - params.entry_percentile)
        out["log_return"] = price.diff()
        out["annual_vol"] = out["log_return"].rolling(params.vol_window).std(ddof=1) * np.sqrt(252.0)
        out["target_size"] = (params.target_vol / out["annual_vol"].replace(0.0, np.nan)).clip(
            lower=0.000001, upper=0.95).fillna(0.95)
        return out

    def generate_signals(
        self, data: pd.DataFrame, indicators: pd.DataFrame,
        ctx: StrategyContext, params: TopologyParams
    ) -> SignalFrame:
        topology = indicators["topology"]
        upper = indicators["upper_threshold"]
        lower = indicators["lower_threshold"]
        ready = indicators[["topology", "upper_threshold", "lower_threshold", "annual_vol"]].notna().all(axis=1)
        ready &= upper > lower
        setup = pd.Series(0, index=data.index, dtype=int)
        setup.loc[ready & (topology > upper)] = 1
        setup.loc[ready & (topology < lower)] = -1
        position = 0
        held = np.zeros(len(data), dtype=int)
        for bar, candidate in enumerate(setup.to_numpy()):
            if position == 0:
                position = int(candidate)
            elif candidate == -position:
                position = 0
            held[bar] = position
        df = pd.DataFrame(index=data.index)
        df["signal"] = held
        df["size"] = indicators["target_size"].shift(1).fillna(0.95).astype(float)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


if __name__ == "__main__":
    strategy = GeneratedStrategy()
    params = TopologyParams()
    index = pd.date_range("2020-01-01", periods=30, freq="D")
    close = pd.Series(np.exp(0.001 * np.arange(30, dtype=float) ** 2), index=index)
    data = pd.DataFrame({"open": close, "high": close, "low": close, "close": close, "volume": 1000.0})
    features = strategy.indicators(data, params)
    assert features["upper_degree"].iloc[-1] == params.graph_window - 1
    assert features["lower_degree"].iloc[-1] == 1
    probe = pd.DataFrame({
        "topology": [0, 1, 0, -1, -1, 0, 1, 1],
        "upper_threshold": 0.5, "lower_threshold": -0.5,
        "annual_vol": 0.2, "target_size": 0.6
    }, index=index[:8])
    result = strategy.generate_signals(data.iloc[:8], probe, None, params).data
    assert result["signal"].tolist() == [0, 0, 1, 1, 0, -1, -1, 0]
    assert result["size"].gt(0).all()
