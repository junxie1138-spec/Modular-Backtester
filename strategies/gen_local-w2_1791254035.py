from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class TopologyParams:
    graph_window: int = 18
    regime_span: int = 9
    reference_window: int = 126
    entry_quantile: float = 0.75


class GeneratedStrategy(BaseStrategy[TopologyParams]):
    strategy_id = "gen_local-w2_1791254035"

    @classmethod
    def params_type(cls):
        return TopologyParams

    def warmup_bars(self, params: TopologyParams) -> int:
        return int(params.graph_window + 2 * params.regime_span + params.reference_window)

    def indicators(self, data: pd.DataFrame, params: TopologyParams) -> pd.DataFrame:
        if min(params.graph_window, params.regime_span, params.reference_window) < 2:
            raise ValueError("Lookbacks must be at least two.")
        if not 0.0 < params.entry_quantile < 1.0:
            raise ValueError("entry_quantile must be between zero and one.")
        close = data["close"].astype(float)
        close = close.where(np.isfinite(close) & (close > 0.0))
        upper = pd.Series(0.0, index=data.index)
        lower = upper.copy()
        ceiling = close.shift(1)
        floor = ceiling.copy()
        # Each endpoint connects only to past nodes; adjacent edges carry no information.
        for lag in range(2, params.graph_window):
            left = close.shift(lag)
            upper += (np.minimum(close, left) > ceiling).astype(float)
            lower += (np.maximum(close, left) < floor).astype(float)
            ceiling = np.maximum(ceiling, left)
            floor = np.minimum(floor, left)
        valid = close.rolling(params.graph_window).count().eq(params.graph_window)
        upper = upper.where(valid)
        lower = lower.where(valid)
        total = upper + lower
        balance = ((upper - lower) / total.replace(0.0, np.nan)).where(total != 0.0, 0.0)
        balance = balance.where(valid)
        regime = balance.rolling(params.regime_span).mean()
        change = regime.diff(params.regime_span)
        magnitude = change.abs()
        threshold = magnitude.shift(1).rolling(params.reference_window).quantile(params.entry_quantile)
        return pd.DataFrame({
            "upper_degree": upper,
            "lower_degree": lower,
            "hub_balance": balance,
            "regime": regime,
            "topology_change": change,
            "change_magnitude": magnitude,
            "entry_threshold": threshold,
        }, index=data.index)

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: TopologyParams,
    ) -> SignalFrame:
        change = indicators["topology_change"]
        threshold = indicators["entry_threshold"]
        events = np.where(change > threshold, 1, np.where(change < -threshold, -1, 0))
        positions = np.zeros(len(data), dtype=int)
        position = 0
        for bar, event in enumerate(events):
            if event:
                if position == 0:
                    position = int(event)
                elif event != position:
                    position = 0
            positions[bar] = position
        df = pd.DataFrame({"signal": positions, "size": 1.0}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check() -> None:
    strategy = GeneratedStrategy()
    params = TopologyParams()
    index = pd.date_range("2020-01-01", periods=400, freq="B")
    close = 100.0 * np.exp(np.cumsum(np.random.default_rng(7).normal(0.0, 0.01, len(index))))
    data = pd.DataFrame({"open": close, "high": close * 1.01, "low": close * 0.99,
                         "close": close, "volume": 1000000.0}, index=index)
    ctx = StrategyContext(symbol="SPY", timeframe="1d", warmup_bars=strategy.warmup_bars(params))
    indicators = strategy.indicators(data, params)
    prefix = strategy.indicators(data.iloc[:300], params)
    pd.testing.assert_frame_equal(indicators.iloc[:300], prefix)
    signals = strategy.generate_signals(data, indicators, ctx, params).data
    pd.testing.assert_frame_equal(signals.iloc[:300], strategy.generate_signals(data.iloc[:300], prefix, ctx, params).data)
    assert signals["signal"].isin([-1, 0, 1]).all() and (signals["size"] > 0.0).all()
    assert signals.iloc[:strategy.warmup_bars(params) - 1]["signal"].eq(0).all()
    probe = pd.DataFrame({"topology_change": [np.nan, 2.0, 0.0, -2.0, 0.0, -2.0, 2.0, 0.0],
                          "entry_threshold": 1.0}, index=index[:8])
    result = strategy.generate_signals(data.iloc[:8], probe, ctx, params).data
    assert result["signal"].tolist() == [0, 0, 1, 1, 0, 0, -1, 0]
    tiny = data.iloc[:3].copy()
    tiny["close"] = [3.0, 1.0, 2.0]
    graph = strategy.indicators(tiny, TopologyParams(graph_window=3, regime_span=2, reference_window=2))
    assert graph["upper_degree"].iloc[-1] == 1.0 and graph["lower_degree"].iloc[-1] == 0.0


if __name__ == "__main__":
    _self_check()
