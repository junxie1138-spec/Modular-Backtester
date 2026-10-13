from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class Params:
    graph_window: int = 12
    seasonal_window: int = 504
    forecast_bars: int = 7
    min_samples: int = 12
    seasonal_edge: float = 0.001
    transitivity_cap: float = 0.45
    atr_window: int = 14
    trail_atr: float = 2.5
    hold_bars: int = 8


class GeneratedStrategy(BaseStrategy[Params]):
    strategy_id = "gen_local-w2_1791330712"

    @classmethod
    def params_type(cls) -> type[Params]:
        return Params

    def warmup_bars(self, params: Params) -> int:
        return int(max(params.graph_window, params.seasonal_window + params.forecast_bars, params.atr_window + 1))

    def indicators(self, data: pd.DataFrame, params: Params) -> pd.DataFrame:
        p = params
        if min(p.graph_window, p.seasonal_window, p.forecast_bars, p.min_samples, p.atr_window, p.hold_bars) < 1 or p.graph_window < 3:
            raise ValueError("Lookbacks must be positive and graph_window at least three.")
        if p.trail_atr <= 0 or p.seasonal_edge < 0 or not 0 <= p.transitivity_cap <= 1:
            raise ValueError("Invalid threshold or trailing distance.")
        out = pd.DataFrame(index=data.index)
        close = data["close"].astype(float)
        previous = close.shift(1)
        tr = pd.concat([data["high"] - data["low"], (data["high"] - previous).abs(), (data["low"] - previous).abs()], axis=1).max(axis=1)
        out["atr"] = tr.rolling(p.atr_window, min_periods=p.atr_window).mean()
        out["phase"] = np.minimum((data.index.day - 1) // 7, 3)
        origin_phase = out["phase"].shift(p.forecast_bars)
        realized = close.pct_change(p.forecast_bars, fill_method=None).replace([np.inf, -np.inf], np.nan)
        out["seasonal_mean"] = np.nan
        out["seasonal_samples"] = 0.0
        for phase in range(4):
            observations = realized.where(origin_phase.eq(phase))
            count = observations.notna().astype(float).rolling(p.seasonal_window, min_periods=p.seasonal_window).sum()
            total = observations.fillna(0.0).rolling(p.seasonal_window, min_periods=p.seasonal_window).sum()
            mean = total.div(count.where(count >= p.min_samples))
            out.loc[out["phase"].eq(phase), "seasonal_mean"] = mean
            out.loc[out["phase"].eq(phase), "seasonal_samples"] = count
        values = np.column_stack([np.log(close.where(close > 0)).shift(lag).to_numpy() for lag in range(p.graph_window - 1, -1, -1)])
        valid = np.isfinite(values).all(axis=1)
        # ponytail: O(N * W^3) natural visibility; keep W small, batch if histories grow.
        adjacency = np.zeros((len(data), p.graph_window, p.graph_window), dtype=float)
        for left in range(p.graph_window - 1):
            for right in range(left + 1, p.graph_window):
                visible = valid.copy()
                for middle in range(left + 1, right):
                    line = values[:, left] + (values[:, right] - values[:, left]) * ((middle - left) / (right - left))
                    visible &= values[:, middle] < line
                adjacency[:, left, right] = visible
                adjacency[:, right, left] = visible
        degree = adjacency.sum(axis=2)
        wedges = (degree * (degree - 1)).sum(axis=1)
        triangle_walks = ((adjacency @ adjacency) * adjacency).sum(axis=(1, 2))
        topology = np.divide(triangle_walks, wedges, out=np.zeros(len(data)), where=wedges > 0)
        out["transitivity"] = np.where(valid, topology, np.nan)
        eligible = out["transitivity"].le(p.transitivity_cap) & out["atr"].gt(0) & out["seasonal_mean"].notna()
        out["candidate"] = np.select([eligible & out["seasonal_mean"].gt(p.seasonal_edge), eligible & out["seasonal_mean"].lt(-p.seasonal_edge)], [1, -1], default=0).astype(int)
        return out

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: Params) -> SignalFrame:
        candidate = indicators["candidate"].fillna(0).astype(int)
        confirmed = candidate.where(candidate.eq(candidate.shift(1)), 0).to_numpy()
        atr = indicators["atr"].to_numpy(dtype=float)
        close = data["close"].to_numpy(dtype=float)
        opening = data["open"].to_numpy(dtype=float)
        decisions = np.zeros(len(data), dtype=int)
        position, age = 0, 0
        highest, lowest, stop, short_stop = np.nan, np.nan, np.nan, np.nan
        warmup = self.warmup_bars(params)
        for i in range(len(data)):
            if position:
                age += 1
                if not np.isfinite(close[i]) or not np.isfinite(atr[i]) or atr[i] <= 0:
                    position = 0
                else:
                    # First held bar is the bar after the entry decision.
                    if age == 1:
                        highest = lowest = opening[i] if np.isfinite(opening[i]) else close[i]
                    highest = max(highest, close[i])
                    lowest = min(lowest, close[i])
                    level = highest - params.trail_atr * atr[i]
                    stop = level if not np.isfinite(stop) else max(stop, level)
                    # Shorts also have a ratcheting adverse-move stop.
                    level_short = lowest + params.trail_atr * atr[i]
                    short_stop = level_short if not np.isfinite(short_stop) else min(short_stop, level_short)
                    if close[i] <= stop or (position == -1 and close[i] >= short_stop) or age >= params.hold_bars:
                        position = 0
            elif i >= warmup - 1 and confirmed[i] and np.isfinite(close[i]) and np.isfinite(atr[i]) and atr[i] > 0:
                position = int(confirmed[i])
                age = 0
                highest = lowest = stop = short_stop = np.nan
            decisions[i] = position
        df = pd.DataFrame({"signal": decisions, "size": 1.0}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check() -> None:
    p = Params(graph_window=5, seasonal_window=16, forecast_bars=3, min_samples=2, atr_window=3)
    strategy = GeneratedStrategy()
    w = strategy.warmup_bars(p)
    index = pd.bdate_range("2020-01-01", periods=w + 20)
    close = np.full(len(index), 100.0)
    close[w + 3:] = 90.0
    data = pd.DataFrame({"open": close, "high": close + 1, "low": close - 1, "close": close, "volume": 1000.0}, index=index)
    indicators = pd.DataFrame({"candidate": 0, "atr": 1.0}, index=index)
    indicators.iloc[w:w + 3, 0] = 1
    signals = strategy.generate_signals(data, indicators, None, p).data["signal"]
    assert signals.iloc[w + 1] == 0 and signals.iloc[w + 2] == 1
    assert signals.iloc[w + 3] == 1 and signals.iloc[w + 4] == 0
    assert set(signals.unique()) <= {-1, 0, 1}
    full = strategy.indicators(data, p)
    prefix = strategy.indicators(data.iloc[:-5], p)
    assert np.allclose(full.iloc[:-5].to_numpy(), prefix.to_numpy(), equal_nan=True)


if __name__ == "__main__":
    _self_check()
