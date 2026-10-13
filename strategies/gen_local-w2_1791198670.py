from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class VisibilityParams:
    lookback: int = 12
    min_reach: float = 0.40
    position_ceiling: float = 0.60


class GeneratedStrategy(BaseStrategy[VisibilityParams]):
    strategy_id = "gen_local-w2_1791198670"

    @classmethod
    def params_type(cls):
        return VisibilityParams

    def warmup_bars(self, params: VisibilityParams) -> int:
        return int(params.lookback) + 2

    def indicators(self, data: pd.DataFrame, params: VisibilityParams) -> pd.DataFrame:
        if params.lookback < 3:
            raise ValueError("lookback must be at least 3")
        if not 0.0 <= params.min_reach <= 1.0:
            raise ValueError("min_reach must be between zero and one")
        if not 0.0 <= params.position_ceiling <= 1.0:
            raise ValueError("position_ceiling must be between zero and one")

        window = params.lookback
        close = data["close"].astype(float)
        close = close.where(np.isfinite(close))
        high = data["high"].astype(float)
        high = high.where(np.isfinite(high))
        low = data["low"].astype(float)
        low = low.where(np.isfinite(low))
        blocker = pd.Series(-np.inf, index=data.index)
        degree = pd.Series(0.0, index=data.index)
        edge_span = pd.Series(0.0, index=data.index)

        # A historical vertex is visible when its backward slope exceeds
        # every intervening backward slope: intermediate closes lie below
        # the straight line joining the two endpoint closes.
        for lag in range(1, window + 1):
            slope = (close.shift(lag) - close) / lag
            visible = slope.gt(blocker) & slope.notna()
            degree += visible.astype(float)
            edge_span += lag * visible.astype(float)
            blocker = pd.concat([blocker, slope], axis=1).max(axis=1)

        ready = close.rolling(window + 1).count().eq(window + 1)
        floor = low.rolling(window + 1, min_periods=window + 1).min()
        ceiling = high.rolling(window + 1, min_periods=window + 1).max()
        width = (ceiling - floor).where(ceiling.gt(floor))
        out = pd.DataFrame(index=data.index)
        out["visibility_degree"] = degree.where(ready)
        out["visibility_edge_span"] = edge_span.where(ready)
        out["visibility_reach"] = (edge_span / degree.replace(0.0, np.nan) / window).where(ready)
        out["channel_floor"] = floor
        out["channel_ceiling"] = ceiling
        out["relative_position"] = (close - floor) / width
        out["close_change"] = close.diff()
        out["ready"] = ready
        return out

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: VisibilityParams,
    ) -> SignalFrame:
        eligible = (
            indicators["ready"]
            & indicators["visibility_reach"].ge(params.min_reach)
            & indicators["relative_position"].between(0.0, params.position_ceiling)
            & indicators["close_change"].gt(0.0)
        ).fillna(False).astype(bool)
        # Enter on the second consecutive eligible close. Remain long
        # while eligibility persists; its first reversal exits immediately.
        confirmed = eligible & eligible.shift(1, fill_value=False)
        df = pd.DataFrame(index=data.index)
        df["signal"] = confirmed.astype(int)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = 1.0
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check() -> None:
    strategy = GeneratedStrategy()
    params = VisibilityParams()
    index = pd.date_range("2020-01-01", periods=80, freq="D")
    close = 100.0 + 0.02 * (np.arange(80) - 30) ** 2
    data = pd.DataFrame({"open": close, "high": close + 0.1, "low": close - 0.1, "close": close, "volume": 1000.0}, index=index)
    derived = strategy.indicators(data, params)
    for t in range(params.lookback, len(data)):
        edges = [k for k in range(1, params.lookback + 1) if all(close[t-j] < close[t] + (close[t-k] - close[t]) * j / k for j in range(1, k))]
        assert np.isclose(derived["visibility_reach"].iloc[t], np.mean(edges) / params.lookback)
    flags = np.array([False, True, True, True, False, True, True])
    sample = data.iloc[:len(flags)]
    injected = pd.DataFrame({"ready": True, "visibility_reach": np.where(flags, 0.8, 0.0), "relative_position": 0.3, "close_change": 1.0}, index=sample.index)
    ctx = StrategyContext(symbol="SPY", timeframe="1d", warmup_bars=strategy.warmup_bars(params))
    signals = strategy.generate_signals(sample, injected, ctx, params).data
    assert signals["signal"].tolist() == [0, 0, 0, 1, 1, 0, 0]
    assert signals["size"].gt(0).all()
    flat = data.assign(open=100.0, high=100.0, low=100.0, close=100.0)
    assert strategy.generate_signals(flat, strategy.indicators(flat, params), ctx, params).data["signal"].eq(0).all()


if __name__ == "__main__":
    _self_check()
