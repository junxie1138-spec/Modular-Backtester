from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class ShapeParams:
    shape_bars: int = 12
    history_bars: int = 504
    neighbors: int = 5
    hold_bars: int = 8
    max_distance: float = 0.70
    min_edge: float = 0.004
    min_win_rate: float = 0.60
    profit_target: float = 0.025
    min_size: float = 0.25
    max_size: float = 0.95


class GeneratedStrategy(BaseStrategy[ShapeParams]):
    strategy_id = "gen_local_1790875738"

    @classmethod
    def params_type(cls):
        return ShapeParams

    def warmup_bars(self, params: ShapeParams) -> int:
        return int(params.history_bars + params.shape_bars + params.hold_bars)

    def indicators(self, data: pd.DataFrame, params: ShapeParams) -> pd.DataFrame:
        p = params
        if not (p.shape_bars >= 4 and p.hold_bars >= 1 and p.neighbors >= 2
                and p.history_bars >= p.shape_bars + p.hold_bars * (p.neighbors + 1)
                and 0 < p.max_distance <= 1 and p.min_edge > 0
                and 0.5 <= p.min_win_rate <= 1 and p.profit_target > 0
                and 0 < p.min_size <= p.max_size <= 1):
            raise ValueError("Invalid graph, analog, exit, or sizing parameters")
        close = data["close"].astype(float)
        windows = np.column_stack([
            close.shift(k).to_numpy() for k in range(p.shape_bars - 1, -1, -1)
        ])
        valid = np.isfinite(windows).all(axis=1) & (windows > 0).all(axis=1)
        edges = {}
        # Adjacent vertices always connect; retain only informative longer edges.
        for left in range(p.shape_bars - 2):
            for right in range(left + 2, p.shape_bars):
                interior = np.max(windows[:, left + 1:right], axis=1)
                visible = np.minimum(windows[:, left], windows[:, right]) > interior
                edges[f"edge_{left}_{right}"] = (visible & valid).astype(float)
        out = pd.DataFrame(edges, index=data.index)
        graph = out.to_numpy(dtype=float)
        out["graph_valid"] = valid
        entry_open = data["open"].astype(float).shift(p.hold_bars - 1)
        realized = (close / entry_open.where(entry_open > 0) - 1).replace(
            [np.inf, -np.inf], np.nan
        )
        outcome_valid = realized.notna().to_numpy() & close.gt(0).to_numpy()
        outcomes = realized.fillna(0).to_numpy()
        out["matured_return"] = outcomes
        out["outcome_valid"] = outcome_valid
        forecast = np.zeros(len(data))
        win_rate = np.zeros(len(data))
        distance = np.ones(len(data))
        # ponytail: bounded brute-force matching; index graphs if history grows materially.
        for t in range(self.warmup_bars(p) - 1, len(data)):
            if not valid[t] or graph[t].sum() < 2:
                continue
            # Entire historical outcome precedes the current shape window.
            candidates = np.arange(
                max(p.shape_bars - 1, t - p.history_bars),
                t - p.shape_bars - p.hold_bars + 1,
                p.shape_bars + p.hold_bars,
                dtype=int,
            )
            candidates = candidates[
                valid[candidates] & outcome_valid[candidates + p.hold_bars]
            ]
            if len(candidates) < p.neighbors:
                continue
            union = np.logical_or(graph[candidates], graph[t]).sum(axis=1)
            mismatch = np.logical_xor(graph[candidates], graph[t]).sum(axis=1)
            distances = mismatch / np.maximum(union, 1)
            chosen = np.argsort(distances, kind="stable")[:p.neighbors]
            if distances[chosen].max() > p.max_distance:
                continue
            weights = 1.0 / (0.05 + distances[chosen])
            labels = outcomes[candidates[chosen] + p.hold_bars]
            forecast[t] = np.average(labels, weights=weights)
            win_rate[t] = np.average(labels > 0, weights=weights)
            distance[t] = np.average(distances[chosen], weights=weights)
        out["expected_return"] = forecast
        out["win_rate"] = win_rate
        out["match_distance"] = distance
        out["strength"] = (
            np.clip(forecast / (4 * p.min_edge), 0, 1)
            * np.clip(2 * win_rate - 1, 0, 1)
            * np.clip(1 - distance, 0, 1)
        )
        return out

    def generate_signals(
        self, data: pd.DataFrame, indicators: pd.DataFrame,
        ctx: StrategyContext, params: ShapeParams,
    ) -> SignalFrame:
        p = params
        eligible = (
            indicators["graph_valid"].fillna(False)
            & indicators["expected_return"].ge(p.min_edge)
            & indicators["win_rate"].ge(p.min_win_rate)
            & indicators["match_distance"].le(p.max_distance)
        ).to_numpy()
        strength = indicators["strength"].fillna(0).clip(0, 1).to_numpy()
        opens = data["open"].to_numpy(dtype=float)
        closes = data["close"].to_numpy(dtype=float)
        signals = np.zeros(len(data), dtype=int)
        sizes = np.full(len(data), p.min_size, dtype=float)
        active = False
        entry_bar = -1
        entry_price = np.nan
        held_size = p.min_size
        for t in range(len(data)):
            if active:
                # A prior close decision fills at this bar's open.
                if t == entry_bar:
                    entry_price = opens[t]
                target_hit = (
                    np.isfinite(entry_price) and entry_price > 0
                    and np.isfinite(closes[t])
                    and closes[t] >= entry_price * (1 + p.profit_target)
                )
                deadline = t - entry_bar + 1 >= p.hold_bars
                if target_hit or deadline or not np.isfinite(entry_price) or entry_price <= 0:
                    active = False
                else:
                    signals[t] = 1
                sizes[t] = held_size
            elif t >= self.warmup_bars(p) - 1 and eligible[t]:
                active = True
                entry_bar = t + 1
                entry_price = np.nan
                held_size = p.min_size + (p.max_size - p.min_size) * strength[t]
                signals[t] = 1
                sizes[t] = held_size
        df = pd.DataFrame({"signal": signals, "size": sizes}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = df["size"].shift(1).fillna(p.min_size).astype(float)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check():
    p = ShapeParams(shape_bars=4, history_bars=20, neighbors=2, hold_bars=3)
    strategy = GeneratedStrategy()
    n = strategy.warmup_bars(p) + 8
    index = pd.date_range("2020-01-01", periods=n)
    data = pd.DataFrame(
        {"open": 100.0, "high": 104.0, "low": 99.0, "close": 100.0, "volume": 1000.0},
        index=index,
    )
    decision = strategy.warmup_bars(p) - 1
    features = pd.DataFrame(
        {"graph_valid": True, "expected_return": 0.0, "win_rate": 0.8,
         "match_distance": 0.1, "strength": 0.75}, index=index,
    )
    features.loc[index[decision], "expected_return"] = 0.02
    ctx = StrategyContext(symbol="SPY", timeframe="1d", warmup_bars=decision + 1)
    result = strategy.generate_signals(data, features, ctx, p).data
    assert result["signal"].iloc[decision] == 0
    assert result["signal"].iloc[decision + 1:decision + 4].tolist() == [1, 1, 1]
    assert result["signal"].iloc[decision + 4] == 0
    assert np.isclose(result["size"].iloc[decision + 1], 0.775)
    data.loc[index[decision + 1], "close"] = 103.0
    result = strategy.generate_signals(data, features, ctx, p).data
    assert result["signal"].iloc[decision + 1] == 1
    assert result["signal"].iloc[decision + 2] == 0
    full = strategy.indicators(data, p)
    prefix = strategy.indicators(data.iloc[:-2], p)
    pd.testing.assert_frame_equal(full.iloc[:-2], prefix)
    assert np.isfinite(full.to_numpy(dtype=float)).all()
    assert result["signal"].isin([0, 1]).all() and result["size"].gt(0).all()


if __name__ == "__main__":
    _self_check()
