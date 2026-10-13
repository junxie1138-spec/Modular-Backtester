from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class VisibilitySeasonParams:
    graph_window: int = 20
    horizon: int = 18
    calendar_window: int = 504
    min_month_samples: int = 18
    closure_threshold: float = 0.60
    seasonal_threshold: float = 0.004

    def __post_init__(self):
        if self.graph_window < 3 or self.horizon < 1 or self.calendar_window < 200:
            raise ValueError("Invalid lookback")
        if self.min_month_samples < 1 or not 0 < self.closure_threshold <= 1:
            raise ValueError("Invalid graph or sample threshold")
        if not np.isfinite(self.seasonal_threshold) or self.seasonal_threshold < 0:
            raise ValueError("Invalid seasonal threshold")


class GeneratedStrategy(BaseStrategy[VisibilitySeasonParams]):
    strategy_id = "gen_local_1791337180"

    @classmethod
    def params_type(cls):
        return VisibilitySeasonParams

    def warmup_bars(self, params):
        return int(max(200, params.calendar_window + params.horizon,
                       params.graph_window + params.horizon - 1))

    def indicators(self, data, params):
        close = data["close"].astype(float).where(
            np.isfinite(data["close"]) & (data["close"] > 0)
        )
        out = pd.DataFrame(index=data.index)
        out["ma200"] = close.rolling(200, min_periods=200).mean()
        out["log_close"] = np.log(close)
        closure = pd.Series(np.nan, index=data.index, dtype=float)
        w = params.graph_window
        if len(data) >= w:
            # https://numpy.org/doc/stable/reference/generated/numpy.lib.stride_tricks.sliding_window_view.html
            windows = np.lib.stride_tricks.sliding_window_view(
                out["log_close"].to_numpy(), w
            )
            # ponytail: O(bars * window^3); batch graphs if long intraday histories are required.
            adjacency = np.zeros((len(windows), w, w), dtype=float)
            for i in range(w - 1):
                maximum_slope = np.full(len(windows), -np.inf)
                for j in range(i + 1, w):
                    slope = (windows[:, j] - windows[:, i]) / (j - i)
                    visible = slope > maximum_slope
                    adjacency[:, i, j] = visible
                    adjacency[:, j, i] = visible
                    maximum_slope = np.maximum(maximum_slope, slope)
            degree = adjacency.sum(axis=2)
            ordered_wedges = (degree * (degree - 1)).sum(axis=1)
            closed_wedges = ((adjacency @ adjacency) * adjacency).sum(axis=(1, 2))
            scores = np.divide(closed_wedges, ordered_wedges,
                               out=np.zeros_like(closed_wedges),
                               where=ordered_wedges > 0)
            scores[~np.isfinite(windows).all(axis=1)] = np.nan
            closure.iloc[w - 1:] = scores
        out["triangle_closure"] = closure
        out["closure_mean"] = closure.rolling(
            params.horizon, min_periods=params.horizon
        ).mean()
        out["matured_return"] = close / close.shift(params.horizon) - 1.0
        month = pd.Series(data.index.month, index=data.index)
        out["origin_month"] = month.shift(params.horizon)
        out["seasonal_return"] = np.nan
        out["seasonal_count"] = 0.0
        history_ready = out["matured_return"].rolling(
            params.calendar_window, min_periods=params.calendar_window
        ).count().eq(params.calendar_window)
        for m in range(1, 13):
            eligible = out["origin_month"].eq(m) & out["matured_return"].notna()
            count = eligible.astype(float).rolling(
                params.calendar_window, min_periods=params.calendar_window
            ).sum()
            total = out["matured_return"].where(eligible, 0.0).rolling(
                params.calendar_window, min_periods=params.calendar_window
            ).sum()
            estimate = (total / count.replace(0.0, np.nan)).where(
                history_ready & count.ge(params.min_month_samples)
            )
            current = month.eq(m)
            out.loc[current, "seasonal_return"] = estimate.loc[current]
            out.loc[current, "seasonal_count"] = count.loc[current]
        return out

    def generate_signals(self, data, indicators, ctx, params):
        seasonal = indicators["seasonal_return"]
        topology = indicators["closure_mean"].ge(params.closure_threshold)
        long_entry = topology & seasonal.gt(params.seasonal_threshold) & (
            data["close"] > indicators["ma200"]
        )
        short_entry = topology & seasonal.lt(-params.seasonal_threshold) & (
            data["close"] < indicators["ma200"]
        )
        desired = np.select([long_entry, short_entry], [1, -1], default=0)
        signal = np.zeros(len(data), dtype=int)
        position = 0
        for i, entry in enumerate(desired):
            if position == 0:
                position = int(entry)
            elif entry == -position:
                position = 0
            signal[i] = position
        df = pd.DataFrame({"signal": signal, "size": 0.95}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


if __name__ == "__main__":
    strategy = GeneratedStrategy()
    params = VisibilitySeasonParams()
    index = pd.date_range("2020-01-01", periods=7, freq="B")
    data = pd.DataFrame({"close": 100.0}, index=index)
    features = pd.DataFrame({
        "seasonal_return": [0.01, 0, -0.01, 0, -0.01, 0.01, 0.01],
        "closure_mean": 1.0,
        "ma200": [99, 99, 101, 99, 101, 99, 99],
    }, index=index)
    actual = strategy.generate_signals(data, features, None, params).data
    assert actual["signal"].tolist() == [0, 1, 1, 0, 0, -1, 0]
    assert actual["size"].gt(0).all()
    rng = np.random.default_rng(7)
    prices = 100 * np.exp(np.cumsum(rng.normal(0.0002, 0.01, 600)))
    history = pd.DataFrame({"close": prices},
                           index=pd.date_range("2019-01-01", periods=600, freq="B"))
    full = strategy.indicators(history, params)
    prefix = strategy.indicators(history.iloc[:570], params)
    pd.testing.assert_frame_equal(full.iloc[:570], prefix)
    assert full["triangle_closure"].dropna().between(0, 1).all()
    assert strategy.warmup_bars(params) == 522
