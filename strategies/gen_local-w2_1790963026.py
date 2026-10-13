from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class TopologyParams:
    graph_window: int = 16
    atr_window: int = 14
    entry_strength: float = 0.55
    stop_atr: float = 1.8
    hold_bars: int = 4
    min_size: float = 0.25
    max_size: float = 0.95


class GeneratedStrategy(BaseStrategy[TopologyParams]):
    strategy_id = "gen_local-w2_1790963026"

    @classmethod
    def params_type(cls):
        return TopologyParams

    def warmup_bars(self, params: TopologyParams) -> int:
        return max(params.graph_window + 1, params.atr_window + 1)

    def indicators(self, data: pd.DataFrame, params: TopologyParams) -> pd.DataFrame:
        if params.graph_window < 3 or params.atr_window < 1:
            raise ValueError("Lookback windows must be positive; graph_window >= 3.")
        if not (0 < params.entry_strength <= 1 and params.stop_atr > 0):
            raise ValueError("Invalid entry strength or ATR multiplier.")
        if params.hold_bars < 1 or not (0 < params.min_size <= params.max_size <= 1):
            raise ValueError("Invalid holding period or position sizes.")
        close = data["close"].astype(float)
        previous = close.shift(1)
        tr = pd.concat([
            data["high"] - data["low"],
            (data["high"] - previous).abs(),
            (data["low"] - previous).abs(),
        ], axis=1).max(axis=1)
        atr = tr.rolling(params.atr_window, min_periods=params.atr_window).mean()
        minimum_slope = pd.Series(np.inf, index=data.index)
        degree = pd.Series(0.0, index=data.index)
        span = pd.Series(0.0, index=data.index)
        # A natural visibility edge exists iff every intervening close is
        # strictly below its chord: backward slopes must set a new minimum.
        for lag in range(1, params.graph_window):
            slope = (close - close.shift(lag)) / lag
            visible = slope.notna() & (slope < minimum_slope)
            bullish = visible & (slope > 0)
            degree += bullish.astype(float)
            span = span.where(~bullish, float(lag))
            minimum_slope = pd.concat([minimum_slope, slope], axis=1).min(axis=1)
        ready = close.rolling(params.graph_window).count().eq(params.graph_window)
        degree = (degree / (params.graph_window - 1)).where(ready)
        span = (span / (params.graph_window - 1)).where(ready)
        strength = np.sqrt(degree * span)
        return pd.DataFrame({
            "positive_degree": degree,
            "positive_span": span,
            "strength": strength,
            "atr": atr,
        }, index=data.index)

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: TopologyParams,
    ) -> SignalFrame:
        strength = indicators["strength"]
        entry = (
            (strength >= params.entry_strength)
            & (strength > strength.shift(1))
            & indicators["atr"].gt(0)
        ).fillna(False).to_numpy()
        conviction = (
            params.min_size
            + (params.max_size - params.min_size) * strength.clip(0, 1)
        ).fillna(params.min_size).to_numpy()
        closes = data["close"].to_numpy(dtype=float)
        atrs = indicators["atr"].to_numpy(dtype=float)
        signals = np.zeros(len(data), dtype=int)
        sizes = np.full(len(data), params.min_size, dtype=float)
        active = False
        age = 0
        highest = -np.inf
        stop = -np.inf
        trade_size = params.min_size
        for i in range(len(data)):
            if active:
                age += 1
                if not (np.isfinite(closes[i]) and np.isfinite(atrs[i]) and atrs[i] > 0):
                    active = False
                else:
                    # This bar is in the trade; the entry decision bar was not.
                    highest = max(highest, closes[i])
                    stop = max(stop, highest - params.stop_atr * atrs[i])
                    if closes[i] <= stop or age >= params.hold_bars:
                        active = False
                if active:
                    signals[i] = 1
                    sizes[i] = trade_size
                # An exit decision cannot also reopen on the same close.
                continue
            if entry[i] and np.isfinite(closes[i]):
                active = True
                age = 0
                highest = -np.inf
                stop = -np.inf
                trade_size = float(conviction[i])
                signals[i] = 1
                sizes[i] = trade_size
        df = pd.DataFrame({"signal": signals, "size": sizes}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = df["size"].shift(1).fillna(params.min_size).astype(float)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check() -> None:
    strategy = GeneratedStrategy()
    params = TopologyParams(graph_window=6, atr_window=3)
    x = np.arange(24, dtype=float)
    close = 100 + 0.02 * x * x
    data = pd.DataFrame({
        "open": close, "high": close + 0.1, "low": close - 0.1,
        "close": close, "volume": 1000.0,
    }, index=pd.date_range("2020-01-01", periods=len(x)))
    indicators = strategy.indicators(data, params)
    assert indicators["strength"].iloc[-1] == 1.0
    ctx = StrategyContext("SPY", "1d", strategy.warmup_bars(params))
    controlled = pd.DataFrame({"strength": 0.0, "atr": 1.0}, index=data.index)
    controlled.iloc[6, 0] = 0.8
    data.loc[data.index[8], "close"] = 90.0
    signals = strategy.generate_signals(data, controlled, ctx, params).data
    assert signals["signal"].iloc[6:10].tolist() == [0, 1, 1, 0]
    expected_size = params.min_size + (params.max_size - params.min_size) * 0.8
    assert np.isclose(signals["size"].iloc[7], expected_size)
    controlled.iloc[6, 0] = 0.6
    smaller = strategy.generate_signals(data, controlled, ctx, params).data
    assert smaller["size"].iloc[7] < signals["size"].iloc[7]
    data.loc[data.index[8], "close"] = close[8]
    timed = strategy.generate_signals(data, controlled, ctx, params).data
    assert timed["signal"].iloc[7:12].tolist() == [1, 1, 1, 1, 0]
    prefix = strategy.indicators(data.iloc[:15], params)
    full = strategy.indicators(data, params)
    pd.testing.assert_frame_equal(prefix, full.iloc[:15])
    assert signals["signal"].isin([0, 1]).all()
    assert np.isfinite(signals["size"]).all() and signals["size"].gt(0).all()


if __name__ == "__main__":
    _self_check()
