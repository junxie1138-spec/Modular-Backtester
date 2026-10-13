from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class VisibilityParams:
    graph_window: int = 30
    breakout_window: int = 20
    bridge_share: float = 0.70
    profit_target: float = 0.04
    max_hold: int = 20


class GeneratedStrategy(BaseStrategy[VisibilityParams]):
    strategy_id = "gen_local_1790849716"

    @classmethod
    def params_type(cls):
        return VisibilityParams

    def warmup_bars(self, params: VisibilityParams) -> int:
        return max(params.graph_window, params.breakout_window) + 1

    def indicators(self, data: pd.DataFrame, params: VisibilityParams) -> pd.DataFrame:
        if (params.graph_window < 4 or params.breakout_window < 2
                or not 0 < params.bridge_share <= 1
                or params.profit_target <= 0 or params.max_hold < 1):
            raise ValueError("Invalid visibility strategy parameters")
        close = data["close"].astype(float)
        values = close.to_numpy()
        envelope = np.full(len(data), -np.inf)
        span_mass = np.zeros(len(data), dtype=float)
        bridge_mass = np.zeros(len(data), dtype=float)
        # Natural visibility: every intermediate vertex lies below the edge.
        # Compute all endpoint neighborhoods with vectorized lag comparisons.
        for lag in range(1, params.graph_window + 1):
            slope = (close.shift(lag).to_numpy() - values) / lag
            visible = np.isfinite(slope) & (slope > envelope)
            span_mass += lag * visible
            if lag > params.graph_window // 2:
                bridge_mass += lag * visible
            envelope = np.maximum(envelope, slope)
        valid = pd.Series(np.isfinite(values) & (values > 0), index=data.index)
        ready = valid.rolling(params.graph_window + 1).sum().eq(params.graph_window + 1)
        out = pd.DataFrame(index=data.index)
        out["span_mass"] = np.where(ready, span_mass, np.nan)
        out["bridge_mass"] = np.where(ready, bridge_mass, np.nan)
        out["bridge_share"] = out["bridge_mass"] / out["span_mass"].replace(0.0, np.nan)
        out["prior_high"] = data["high"].shift(1).rolling(params.breakout_window).max()
        out["entry"] = (
            ready
            & out["bridge_share"].ge(params.bridge_share)
            & close.gt(out["prior_high"])
        ).fillna(False)
        return out

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: VisibilityParams,
    ) -> SignalFrame:
        decisions = np.zeros(len(data), dtype=int)
        entries = indicators["entry"].fillna(False).to_numpy(dtype=bool)
        opens = data["open"].to_numpy(dtype=float)
        closes = data["close"].to_numpy(dtype=float)
        active = False
        pending = False
        entry_price = 0.0
        entry_bar = 0
        for i in range(len(data)):
            # The previous close's entry decision fills at this bar's open.
            if pending:
                pending = False
                if np.isfinite(opens[i]) and opens[i] > 0:
                    active = True
                    entry_price = opens[i]
                    entry_bar = i
            if active:
                target_hit = (
                    np.isfinite(closes[i])
                    and closes[i] >= entry_price * (1.0 + params.profit_target)
                )
                timed_out = i - entry_bar + 1 >= params.max_hold
                if target_hit or timed_out:
                    active = False
                else:
                    decisions[i] = 1
            elif entries[i] and np.isfinite(closes[i]) and closes[i] > 0:
                pending = True
                decisions[i] = 1
        df = pd.DataFrame({"signal": decisions, "size": 1.0}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check():
    strategy = GeneratedStrategy()
    index = pd.date_range("2020-01-01", periods=6)
    data = pd.DataFrame(
        {"open": 100.0, "high": 106.0, "low": 99.0,
         "close": [100.0, 100.0, 105.0, 100.0, 100.0, 100.0],
         "volume": 1000.0}, index=index
    )
    indicators = pd.DataFrame({"entry": [True, False, False, False, False, False]}, index=index)
    ctx = StrategyContext(symbol="SPY", timeframe="1d", warmup_bars=0)
    target = strategy.generate_signals(data, indicators, ctx, VisibilityParams())
    assert target.data["signal"].tolist() == [0, 1, 1, 0, 0, 0]
    timed = strategy.generate_signals(
        data, indicators, ctx, VisibilityParams(profit_target=0.5, max_hold=3)
    )
    assert timed.data["signal"].tolist() == [0, 1, 1, 1, 0, 0]
    close = 100.0 + np.arange(60, dtype=float) ** 2
    graph_data = pd.DataFrame(
        {"open": close, "high": close + 0.1, "low": close - 0.1,
         "close": close, "volume": 1000.0},
        index=pd.date_range("2020-01-01", periods=60),
    )
    features = strategy.indicators(graph_data, VisibilityParams())
    assert not features["entry"].iloc[:30].any()
    assert features["entry"].iloc[30:].all()
    prefix = strategy.indicators(graph_data.iloc[:45], VisibilityParams())
    pd.testing.assert_frame_equal(features.iloc[:45], prefix)


if __name__ == "__main__":
    _self_check()
