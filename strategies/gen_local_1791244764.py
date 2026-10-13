from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class Params:
    graph_window: int = 12
    season_window: int = 504
    reach_cutoff: float = 0.35
    min_samples: int = 12
    edge_bps: float = 15.0
    atr_window: int = 14
    stop_atr: float = 1.5
    hold_bars: int = 2


class GeneratedStrategy(BaseStrategy[Params]):
    strategy_id = "gen_local_1791244764"

    @classmethod
    def params_type(cls):
        return Params

    def warmup_bars(self, params: Params) -> int:
        return int(max(200, params.season_window + params.graph_window + params.hold_bars, params.atr_window + 1))

    def indicators(self, data: pd.DataFrame, params: Params) -> pd.DataFrame:
        p = params
        if p.graph_window < 3 or p.season_window < 1 or p.atr_window < 1:
            raise ValueError("Lookbacks must be positive; graph_window must be at least three.")
        if p.hold_bars not in (1, 2) or p.min_samples < 1:
            raise ValueError("hold_bars must be one or two; min_samples must be positive.")
        if not (0.0 < p.reach_cutoff < 1.0) or not np.isfinite(p.stop_atr) or p.stop_atr <= 0:
            raise ValueError("Invalid reach cutoff or ATR stop.")
        if not np.isfinite(p.edge_bps) or p.edge_bps < 0:
            raise ValueError("edge_bps must be finite and nonnegative.")
        out = pd.DataFrame(index=data.index)
        span = (data["high"] - data["low"]).replace(0.0, np.nan)
        out["body_pressure"] = ((data["close"] - data["open"]) / span).replace([np.inf, -np.inf], np.nan)
        x = out["body_pressure"]
        frontier = pd.Series(-np.inf, index=data.index)
        degree = pd.Series(0.0, index=data.index)
        edge_distance = pd.Series(0.0, index=data.index)
        # Natural visibility: a past vertex is visible iff its backward slope
        # strictly exceeds every intervening slope from the current vertex.
        for distance in range(1, p.graph_window):
            slope = (x.shift(distance) - x) / distance
            visible = slope.notna() & slope.gt(frontier)
            degree += visible.astype(float)
            edge_distance += distance * visible.astype(float)
            frontier = pd.concat([frontier, slope], axis=1).max(axis=1)
        complete = x.notna().rolling(p.graph_window, min_periods=p.graph_window).sum().eq(p.graph_window)
        out["degree"] = degree.where(complete)
        out["reach"] = (edge_distance / degree.replace(0.0, np.nan) / (p.graph_window - 1)).where(complete)
        out["state"] = out["reach"].ge(p.reach_cutoff).astype(float).where(complete)
        out["weekday"] = pd.Series(data.index.dayofweek, index=data.index)
        out["realized_return"] = (data["close"] / data["open"].shift(p.hold_bars - 1) - 1.0).replace([np.inf, -np.inf], np.nan)
        out["origin_weekday"] = out["weekday"].shift(p.hold_bars)
        out["origin_state"] = out["state"].shift(p.hold_bars)
        out["seasonal_mean"] = np.nan
        out["sample_count"] = 0.0
        for weekday in range(5):
            for state in (0, 1):
                historical = out["origin_weekday"].eq(weekday) & out["origin_state"].eq(state) & out["realized_return"].notna()
                label = f"w{weekday}_s{state}"
                out[f"count_{label}"] = historical.astype(float).rolling(p.season_window, min_periods=p.season_window).sum()
                out[f"sum_{label}"] = out["realized_return"].where(historical, 0.0).rolling(p.season_window, min_periods=p.season_window).sum()
                estimate = out[f"sum_{label}"] / out[f"count_{label}"].replace(0.0, np.nan)
                current = out["weekday"].eq(weekday) & out["state"].eq(state)
                out.loc[current, "sample_count"] = out.loc[current, f"count_{label}"]
                out.loc[current, "seasonal_mean"] = estimate.where(out[f"count_{label}"].ge(p.min_samples)).loc[current]
        previous_close = data["close"].shift(1)
        out["true_range"] = pd.concat([
            data["high"] - data["low"],
            (data["high"] - previous_close).abs(),
            (data["low"] - previous_close).abs(),
        ], axis=1).max(axis=1).where(previous_close.notna())
        out["atr"] = out["true_range"].rolling(p.atr_window, min_periods=p.atr_window).mean()
        out["ma200"] = data["close"].rolling(200, min_periods=200).mean()
        return out

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: Params) -> SignalFrame:
        threshold = params.edge_bps / 10000.0
        candidate = pd.Series(0, index=data.index, dtype=int)
        candidate.loc[indicators["seasonal_mean"].gt(threshold) & data["close"].gt(indicators["ma200"])] = 1
        candidate.loc[indicators["seasonal_mean"].lt(-threshold) & data["close"].lt(indicators["ma200"])] = -1
        ready = np.arange(len(data)) >= self.warmup_bars(params) - 1
        valid = np.isfinite(indicators["atr"].to_numpy()) & indicators["atr"].gt(0).to_numpy()
        candidate = candidate.where(ready & valid, 0).to_numpy()
        opens = data["open"].to_numpy(dtype=float)
        closes = data["close"].to_numpy(dtype=float)
        atr = indicators["atr"].to_numpy(dtype=float)
        targets = np.zeros(len(data), dtype=int)
        position = 0
        age = 0
        stop = np.nan
        for i in range(len(data)):
            # Yesterday's target fills at today's open; freeze yesterday's ATR.
            if position == 0 and i > 0 and targets[i - 1] != 0:
                if np.isfinite(opens[i]) and opens[i] > 0:
                    position = int(targets[i - 1])
                    stop = opens[i] - position * params.stop_atr * atr[i - 1]
                    age = 0
                else:
                    continue
            if position != 0:
                age += 1
                stopped = not np.isfinite(closes[i]) or position * (closes[i] - stop) < 0
                if stopped or age >= params.hold_bars:
                    position = 0
                    age = 0
                    # Leave a flat target for the next open before any reentry.
                    continue
                targets[i] = position
            else:
                targets[i] = candidate[i]
        df = pd.DataFrame({"signal": targets, "size": 1.0}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


if __name__ == "__main__":
    p = Params(season_window=100, min_samples=3)
    strategy = GeneratedStrategy()
    index = pd.bdate_range("2020-01-01", periods=450)
    rng = np.random.default_rng(7)
    close = 100.0 * np.exp(np.cumsum(rng.normal(0.0002, 0.01, len(index))))
    opening = np.r_[close[0], close[:-1]]
    data = pd.DataFrame({"open": opening, "high": np.maximum(opening, close) + 1.0, "low": np.minimum(opening, close) - 1.0, "close": close, "volume": 1000000.0}, index=index)
    ctx = StrategyContext("SPY", "1d", strategy.warmup_bars(p))
    ind = strategy.indicators(data, p)
    result = strategy.generate_signals(data, ind, ctx, p).data
    prefix = data.iloc[:350]
    prefix_result = strategy.generate_signals(prefix, strategy.indicators(prefix, p), ctx, p).data
    assert result.iloc[:350].equals(prefix_result)
    assert result["signal"].isin([-1, 0, 1]).all() and result["size"].gt(0).all()
    assert result["signal"].iloc[:strategy.warmup_bars(p)].eq(0).all()
    w = strategy.warmup_bars(p)
    fixture = pd.DataFrame({"open": 100.0, "high": 101.0, "low": 95.0, "close": 100.0, "volume": 1000.0}, index=index[:w + 4])
    for direction in (1, -1):
        fixture["close"] = 100.0
        fixture.iloc[w, fixture.columns.get_loc("close")] = 100.0 - direction * 4.0
        fake = pd.DataFrame({"seasonal_mean": direction * 0.02, "ma200": 100.0 - direction, "atr": 2.0}, index=fixture.index)
        stopped = strategy.generate_signals(fixture, fake, ctx, p).data["signal"]
        assert stopped.iloc[w] == direction and stopped.iloc[w + 1] == 0
    print("Self-check passed.")
