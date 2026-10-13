from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class VisibilityParams:
    graph_window: int = 16
    reach_threshold: float = 0.35
    atr_window: int = 14
    breakeven_pct: float = 0.012
    trail_atr: float = 2.0
    max_hold: int = 10


class GeneratedStrategy(BaseStrategy[VisibilityParams]):
    strategy_id = "gen_local-w2_1791249246"

    @classmethod
    def params_type(cls):
        return VisibilityParams

    def warmup_bars(self, params: VisibilityParams) -> int:
        return max(params.graph_window + 1, params.atr_window + 1)

    def indicators(self, data: pd.DataFrame, params: VisibilityParams) -> pd.DataFrame:
        if params.graph_window < 3 or params.atr_window < 1:
            raise ValueError("Lookbacks must be graph_window >= 3 and atr_window >= 1.")
        if not (0 < params.reach_threshold <= 1 and params.breakeven_pct > 0):
            raise ValueError("Reach must be in (0, 1] and breakeven_pct positive.")
        if params.trail_atr <= 0 or params.max_hold < 1:
            raise ValueError("trail_atr and max_hold must be positive.")
        close = data["close"].where(np.isfinite(data["close"]) & (data["close"] > 0))
        n = len(data)
        minimum_slope = np.full(n, np.inf)
        degree = np.zeros(n)
        upward = np.zeros(n)
        downward = np.zeros(n)
        # Natural visibility: all intermediate prices lie strictly below the edge.
        # Backward edges are precisely new minimum slopes from the newest node.
        for lag in range(1, params.graph_window):
            slope = ((close - close.shift(lag)) / lag).to_numpy()
            visible = np.isfinite(slope) & (slope < minimum_slope)
            degree += visible
            upward += lag * (visible & (slope > 0))
            downward += lag * (visible & (slope < 0))
            minimum_slope = np.minimum(minimum_slope, slope)
        valid = close.rolling(params.graph_window, min_periods=params.graph_window).count().eq(params.graph_window)
        denominator = params.graph_window * (params.graph_window - 1) / 2.0
        result = pd.DataFrame(index=data.index)
        result["visible_degree"] = pd.Series(degree, index=data.index).where(valid)
        result["upward_reach"] = pd.Series(upward / denominator, index=data.index).where(valid)
        result["downward_reach"] = pd.Series(downward / denominator, index=data.index).where(valid)
        previous = close.shift(1)
        tr = pd.concat([
            data["high"] - data["low"],
            (data["high"] - previous).abs(),
            (data["low"] - previous).abs(),
        ], axis=1).max(axis=1)
        result["atr"] = tr.rolling(params.atr_window, min_periods=params.atr_window).mean()
        result["candidate"] = (
            result["upward_reach"].ge(params.reach_threshold)
            & result["downward_reach"].le(0.05)
            & result["atr"].gt(0)
        ).fillna(False)
        return result

    def generate_signals(
        self, data: pd.DataFrame, indicators: pd.DataFrame,
        ctx: StrategyContext, params: VisibilityParams,
    ) -> SignalFrame:
        confirmed = (
            indicators["candidate"].fillna(False)
            & indicators["candidate"].shift(1, fill_value=False)
        ).to_numpy(dtype=bool)
        prices = data[["open", "high", "low", "close"]].to_numpy(dtype=float)
        atr = indicators["atr"].to_numpy(dtype=float)
        signals = np.zeros(len(data), dtype=int)
        active = False
        entry = np.nan
        stop = np.nan
        entry_bar = -1
        armed = False
        for i in range(len(data)):
            if not np.isfinite(prices[i]).all() or (prices[i] <= 0).any() or not np.isfinite(atr[i]) or atr[i] <= 0:
                active = False
                entry = np.nan
                continue
            if active:
                if np.isnan(entry):
                    # Yesterday's decision fills at today's open.
                    entry = prices[i, 0]
                    entry_bar = i
                    stop = entry - params.trail_atr * atr[i - 1]
                    armed = False
                # A stop touch is observed today; flattening executes next open.
                # Test the previously known stop before advancing today's ratchet.
                if prices[i, 2] <= stop or i - entry_bar + 1 >= params.max_hold:
                    active = False
                    entry = np.nan
                    continue
                if prices[i, 3] >= entry * (1.0 + params.breakeven_pct):
                    armed = True
                if armed:
                    stop = max(stop, entry, prices[i, 3] - params.trail_atr * atr[i])
                signals[i] = 1
            elif i >= self.warmup_bars(params) - 1 and confirmed[i]:
                active = True
                entry = np.nan
                signals[i] = 1
        df = pd.DataFrame({"signal": signals, "size": 0.95}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check():
    strategy = GeneratedStrategy()
    params = VisibilityParams(graph_window=6, atr_window=3)
    t = np.arange(30, dtype=float)
    close = 100.0 + 0.1 * t + 0.01 * t * t
    data = pd.DataFrame({
        "open": close - 0.05, "high": close + 0.2,
        "low": close - 0.2, "close": close, "volume": 1000000.0,
    }, index=pd.date_range("2020-01-01", periods=len(t)))
    indicators = strategy.indicators(data, params)
    assert np.allclose(indicators["upward_reach"].iloc[5:], 1.0)
    signals = strategy.generate_signals(data, indicators, None, params).data
    assert signals["signal"].iloc[:7].eq(0).all()
    assert signals["signal"].iloc[7] == 1
    assert signals["signal"].isin([0, 1]).all() and signals["size"].gt(0).all()
    altered = data.copy()
    altered.iloc[9, altered.columns.get_loc("low")] = 1.0
    exited = strategy.generate_signals(altered, strategy.indicators(altered, params), None, params).data
    assert exited["signal"].iloc[9] == 1 and exited["signal"].iloc[10] == 0
    prefix = strategy.generate_signals(data.iloc[:12], strategy.indicators(data.iloc[:12], params), None, params).data
    pd.testing.assert_frame_equal(prefix, signals.iloc[:12])


if __name__ == "__main__":
    _self_check()
