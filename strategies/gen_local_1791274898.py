from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class Params:
    window: int = 12
    library_bars: int = 504
    neighbors: int = 8
    hold_bars: int = 8
    entry_threshold: float = 0.35
    max_distance: float = 0.20
    min_size: float = 0.20
    max_size: float = 0.95


class GeneratedStrategy(BaseStrategy[Params]):
    strategy_id = "gen_local_1791274898"

    @classmethod
    def params_type(cls):
        return Params

    def warmup_bars(self, params: Params) -> int:
        return int(params.library_bars + 2 * params.window + params.hold_bars)

    def indicators(self, data: pd.DataFrame, params: Params) -> pd.DataFrame:
        p = params
        if (p.window < 3 or p.hold_bars < 1 or p.neighbors < 2
                or p.library_bars < p.neighbors * (p.window + p.hold_bars)
                or not 0 < p.entry_threshold <= 1
                or not 0 < p.max_distance <= 1
                or not 0 < p.min_size <= p.max_size <= 1):
            raise ValueError("Invalid visibility-analog parameters")
        n, w = len(data), p.window
        names = []
        columns = []
        if n >= w:
            windows = np.lib.stride_tricks.sliding_window_view(
                data["close"].to_numpy(dtype=float), w
            )
            valid = np.isfinite(windows).all(axis=1) & (windows > 0).all(axis=1)
        for left in range(w - 2):
            for right in range(left + 2, w):
                for side in ("upper", "lower"):
                    values = np.full(n, np.nan)
                    if n >= w:
                        interior = windows[:, left + 1:right]
                        if side == "upper":
                            visible = interior.max(axis=1) < np.minimum(
                                windows[:, left], windows[:, right]
                            )
                        else:
                            visible = interior.min(axis=1) > np.maximum(
                                windows[:, left], windows[:, right]
                            )
                        values[w - 1:] = np.where(valid, visible.astype(float), np.nan)
                    names.append(f"{side}_{left}_{right}")
                    columns.append(values)
        graph = np.column_stack(columns)
        out = pd.DataFrame(graph, index=data.index, columns=names)
        opens = data["open"].where(data["open"] > 0)
        out["realized_return"] = (
            opens / opens.shift(p.hold_bars) - 1.0
        ).replace([np.inf, -np.inf], np.nan)
        realized = out["realized_return"].to_numpy()
        forecast = np.zeros(n)
        conviction = np.zeros(n)
        distance = np.ones(n)
        count = np.zeros(n, dtype=int)
        # ponytail: exhaustive search over a bounded library; index graphs if it grows.
        for t in range(self.warmup_bars(p) - 1, n):
            if not np.isfinite(graph[t]).all():
                continue
            # Historical outcomes end before the current graph window begins.
            end = t - w - p.hold_bars
            candidates = np.arange(max(w - 1, end - p.library_bars), end)
            labels = realized[candidates + p.hold_bars + 1]
            usable = np.isfinite(labels) & np.isfinite(graph[candidates]).all(axis=1)
            candidates, labels = candidates[usable], labels[usable]
            if len(candidates) < p.neighbors:
                continue
            distances = np.mean(graph[candidates] != graph[t], axis=1)
            selected = []
            for j in np.argsort(distances, kind="stable"):
                if distances[j] > p.max_distance:
                    break
                if all(abs(int(candidates[j] - candidates[k])) >= w + p.hold_bars
                       for k in selected):
                    selected.append(int(j))
                if len(selected) == p.neighbors:
                    break
            count[t] = len(selected)
            if len(selected) < p.neighbors:
                continue
            outcomes = labels[selected]
            forecast[t] = float(outcomes.mean())
            conviction[t] = float(np.clip(
                forecast[t] / max(float(np.abs(outcomes).mean()), 1e-12), -1.0, 1.0
            ))
            distance[t] = float(distances[selected].mean())
        out["forecast"] = forecast
        out["conviction"] = conviction
        out["match_distance"] = distance
        out["analog_count"] = count
        return out

    def generate_signals(
        self, data: pd.DataFrame, indicators: pd.DataFrame,
        ctx: StrategyContext, params: Params,
    ) -> SignalFrame:
        n = len(data)
        signals = np.zeros(n, dtype=int)
        sizes = np.full(n, params.min_size, dtype=float)
        scores = indicators["conviction"].fillna(0).to_numpy(dtype=float)
        counts = indicators["analog_count"].fillna(0).to_numpy(dtype=float)
        direction, expiry, entry_size = 0, -1, params.min_size
        for t in range(self.warmup_bars(params) - 1, n):
            if direction != 0:
                if t == expiry:
                    direction = 0
                    continue
            elif (np.isfinite(scores[t]) and counts[t] >= params.neighbors
                  and abs(scores[t]) >= params.entry_threshold):
                direction = 1 if scores[t] > 0 else -1
                expiry = t + params.hold_bars
                entry_size = float(np.clip(
                    params.min_size + (params.max_size - params.min_size) * abs(scores[t]),
                    params.min_size, params.max_size,
                ))
            signals[t] = direction
            if direction:
                sizes[t] = entry_size
        df = pd.DataFrame({"signal": signals, "size": sizes}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = df["size"].shift(1).fillna(params.min_size).astype(float)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


if __name__ == "__main__":
    p = Params(window=8, library_bars=96, neighbors=4, hold_bars=6)
    strategy = GeneratedStrategy()
    x = np.arange(220, dtype=float)
    close = 100.0 * np.exp(0.025 * np.sin(x / 3.0) + 0.008 * np.sin(x / 7.0))
    data = pd.DataFrame(
        {"open": close, "high": close * 1.01, "low": close * 0.99,
         "close": close, "volume": np.full(len(x), 1000000.0)},
        index=pd.date_range("2020-01-01", periods=len(x), freq="B"),
    )
    ind = strategy.indicators(data, p)
    prefix = strategy.indicators(data.iloc[:180], p)
    assert np.allclose(ind.iloc[:180].to_numpy(), prefix.to_numpy(), equal_nan=True)
    assert ind["analog_count"].max() >= p.neighbors
    forced = ind.copy()
    forced["conviction"] = 0.8
    forced["analog_count"] = p.neighbors
    start = strategy.warmup_bars(p)
    forced.iloc[start + 1, forced.columns.get_loc("conviction")] = -0.8
    ctx = StrategyContext(symbol="SPY", timeframe="1d", warmup_bars=start)
    result = strategy.generate_signals(data, forced, ctx, p).data
    assert (result["signal"].iloc[:start] == 0).all()
    assert (result["signal"].iloc[start:start + p.hold_bars] == 1).all()
    assert result["signal"].iloc[start + p.hold_bars] == 0
    assert result["size"].iloc[start:start + p.hold_bars].nunique() == 1
    forced["conviction"] = -0.8
    short = strategy.generate_signals(data, forced, ctx, p).data
    assert (short["signal"].iloc[start:start + p.hold_bars] == -1).all()
    assert result["signal"].isin([-1, 0, 1]).all()
    assert np.isfinite(result["size"]).all() and (result["size"] > 0).all()
