from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class VisibilityParams:
    lookback: int = 20
    graph_threshold: float = 0.30
    edge_fraction: float = 0.40
    profit_target: float = 0.025
    max_hold_bars: int = 8


class GeneratedStrategy(BaseStrategy[VisibilityParams]):
    strategy_id = "gen_local_1791291813"

    @classmethod
    def params_type(cls):
        return VisibilityParams

    @staticmethod
    def warmup_bars(params: VisibilityParams) -> int:
        return int(params.lookback)

    @staticmethod
    def indicators(data: pd.DataFrame, params: VisibilityParams) -> pd.DataFrame:
        w = params.lookback
        if w < 3 or not 0 < params.graph_threshold <= 1:
            raise ValueError("Require lookback >= 3 and graph_threshold in (0, 1].")
        if not 0 < params.edge_fraction < 0.5:
            raise ValueError("Require edge_fraction in (0, 0.5).")
        close = data["close"].astype(float).replace([np.inf, -np.inf], np.nan)
        log_close = np.log(close.where(close > 0))
        valid = log_close.rolling(w, min_periods=w).count().eq(w)
        minimum_slope = log_close - log_close.shift(1)
        maximum_slope = minimum_slope.copy()
        upper_degree = pd.Series(0.0, index=data.index)
        lower_degree = upper_degree.copy()
        # Endpoint natural visibility: every intervening point must lie below its chord.
        # Negating log prices gives the mirrored graph; record slopes find both degrees.
        for lag in range(2, w):
            slope = (log_close - log_close.shift(lag)) / lag
            upper_degree += slope.lt(minimum_slope).astype(float)
            lower_degree += slope.gt(maximum_slope).astype(float)
            minimum_slope = np.minimum(minimum_slope, slope)
            maximum_slope = np.maximum(maximum_slope, slope)
        upper_degree = upper_degree.where(valid)
        lower_degree = lower_degree.where(valid)
        graph_bias = (upper_degree - lower_degree) / (w - 2)
        range_low = close.rolling(w, min_periods=w).min()
        range_high = close.rolling(w, min_periods=w).max()
        range_position = (close - range_low) / (range_high - range_low).replace(0, np.nan)
        buy = valid & graph_bias.ge(params.graph_threshold) & range_position.le(params.edge_fraction)
        sell = valid & graph_bias.le(-params.graph_threshold) & range_position.ge(1 - params.edge_fraction)
        candidate = pd.Series(np.select([buy, sell], [1, -1], default=0), index=data.index, dtype=int)
        return pd.DataFrame({
            "log_close": log_close,
            "upper_degree": upper_degree,
            "lower_degree": lower_degree,
            "graph_bias": graph_bias,
            "range_low": range_low,
            "range_high": range_high,
            "range_position": range_position,
            "candidate": candidate,
        }, index=data.index)

    @staticmethod
    def generate_signals(data: pd.DataFrame, indicators: pd.DataFrame,
                         ctx: StrategyContext, params: VisibilityParams) -> SignalFrame:
        if params.max_hold_bars < 1 or not np.isfinite(params.profit_target) or params.profit_target <= 0:
            raise ValueError("Require positive profit_target and max_hold_bars >= 1.")
        candidate = indicators["candidate"].fillna(0).astype(int)
        confirmed = candidate.where(candidate.eq(candidate.shift(1)), 0).to_numpy()
        opens = data["open"].to_numpy(dtype=float)
        closes = data["close"].to_numpy(dtype=float)
        decisions = np.zeros(len(data), dtype=int)
        side = 0
        entry_bar = -1
        entry_price = np.nan
        for i in range(len(data)):
            if side:
                # This open is observed only after the preceding close's entry decision.
                if i == entry_bar:
                    entry_price = opens[i]
                if not np.isfinite(entry_price) or entry_price <= 0 or not np.isfinite(closes[i]) or closes[i] <= 0:
                    side = 0
                elif side * (closes[i] / entry_price - 1) >= params.profit_target or i - entry_bar + 1 >= params.max_hold_bars:
                    side = 0
            elif confirmed[i] and np.isfinite(closes[i]) and closes[i] > 0:
                side = int(confirmed[i])
                entry_bar = i + 1
                entry_price = np.nan
            decisions[i] = side
        df = pd.DataFrame({"signal": decisions, "size": 0.95}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check():
    index = pd.date_range("2020-01-01", periods=8)
    data = pd.DataFrame({"open": 100.0, "high": 105.0, "low": 95.0,
                         "close": 100.0, "volume": 1000.0}, index=index)
    params = VisibilityParams()
    for side in (1, -1):
        signals = pd.DataFrame({"candidate": [side, side, 0, 0, 0, 0, 0, 0]}, index=index)
        prices = data.copy()
        prices.loc[index[3], "close"] = 104.0 if side == 1 else 96.0
        result = GeneratedStrategy.generate_signals(prices, signals, None, params).data
        assert result["signal"].tolist() == [0, 0, side, side, 0, 0, 0, 0]
        timed = GeneratedStrategy.generate_signals(data, signals, None, VisibilityParams(max_hold_bars=2)).data
        assert timed["signal"].tolist() == [0, 0, side, side, 0, 0, 0, 0]
    interrupted = pd.DataFrame({"candidate": [1, 0, 1, 0, -1, 0, -1, 0]}, index=index)
    assert not GeneratedStrategy.generate_signals(data, interrupted, None, params).data["signal"].any()
    x = np.arange(60, dtype=float)
    curve = pd.DataFrame({"close": np.exp(0.001 * x * x)}, index=pd.date_range("2021-01-01", periods=60))
    graph = GeneratedStrategy.indicators(curve, params)
    assert graph["candidate"].iloc[:params.lookback - 1].eq(0).all()
    assert np.allclose(graph["graph_bias"].iloc[params.lookback - 1:], 1.0)
    pd.testing.assert_frame_equal(graph.iloc[:40], GeneratedStrategy.indicators(curve.iloc[:40], params))


if __name__ == "__main__":
    _self_check()
