from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class TopologyParams:
    graph_window: int = 12
    percentile_window: int = 126
    closure_quantile: float = 0.75
    profit_target: float = 0.025
    max_hold: int = 8


class GeneratedStrategy(BaseStrategy[TopologyParams]):
    strategy_id = "gen_local_1790973755"

    @classmethod
    def params_type(cls):
        return TopologyParams

    def warmup_bars(self, params: TopologyParams) -> int:
        return params.graph_window + params.percentile_window + 2

    def indicators(self, data: pd.DataFrame, params: TopologyParams) -> pd.DataFrame:
        if params.graph_window < 4 or params.percentile_window < 2:
            raise ValueError("Graph window must be >= 4 and percentile window >= 2.")
        if not 0.0 < params.closure_quantile < 1.0:
            raise ValueError("Closure quantile must lie strictly between zero and one.")
        if params.profit_target <= 0.0 or params.max_hold < 1:
            raise ValueError("Profit target and holding period must be positive.")

        out = pd.DataFrame(index=data.index)
        close = data["close"].where(data["close"] > 0.0)
        out["log_return"] = np.log(close).diff().replace([np.inf, -np.inf], np.nan)
        out["amplitude"] = out["log_return"].abs()
        out["triangle_closure"] = np.nan
        w = params.graph_window
        if len(data) >= w:
            windows = np.lib.stride_tricks.sliding_window_view(
                out["amplitude"].to_numpy(dtype=float), w
            )
            valid = np.isfinite(windows).all(axis=1)
            # ponytail: dense O(bars * window^2) memory; batch windows for large histories.
            adjacency = np.zeros((len(windows), w, w), dtype=np.float64)
            for left in range(w - 1):
                distances = np.arange(1, w - left, dtype=float)
                slopes = (windows[:, left + 1:] - windows[:, left, None]) / distances
                previous = np.concatenate(
                    [np.full((len(windows), 1), -np.inf),
                     np.maximum.accumulate(slopes, axis=1)[:, :-1]], axis=1
                )
                visible = slopes > previous
                adjacency[:, left, left + 1:] = visible
                adjacency[:, left + 1:, left] = visible
            degree = adjacency.sum(axis=2)
            wedges = (degree * (degree - 1.0) / 2.0).sum(axis=1)
            # Each triangle contributes six ordered two-edge paths.
            closed_paths = ((adjacency @ adjacency) * adjacency).sum(axis=(1, 2))
            closure = np.divide(
                closed_paths, 2.0 * wedges,
                out=np.zeros(len(windows), dtype=float), where=wedges > 0.0
            )
            closure[~valid] = np.nan
            out.iloc[w - 1:, out.columns.get_loc("triangle_closure")] = closure

        out["closure_threshold"] = out["triangle_closure"].shift(1).rolling(
            params.percentile_window, min_periods=params.percentile_window
        ).quantile(params.closure_quantile)
        out["drift"] = out["log_return"].rolling(w, min_periods=w).sum()
        crossing = (
            (out["triangle_closure"] > out["closure_threshold"])
            & (out["triangle_closure"].shift(1) <= out["closure_threshold"].shift(1))
        )
        direction = np.sign(out["drift"].fillna(0.0))
        out["entry_direction"] = direction.where(crossing, 0).astype(int)
        return out

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: TopologyParams,
    ) -> SignalFrame:
        if params.profit_target <= 0.0 or params.max_hold < 1:
            raise ValueError("Profit target and holding period must be positive.")
        df = pd.DataFrame(index=data.index)
        decisions = np.zeros(len(data), dtype=int)
        entries = indicators["entry_direction"].fillna(0).to_numpy(dtype=int)
        opens = data["open"].to_numpy(dtype=float)
        closes = data["close"].to_numpy(dtype=float)
        side = 0
        entry_bar = -1
        entry_price = np.nan
        for i in range(len(data)):
            if side:
                # The decision was made yesterday; today's open establishes cost.
                if i == entry_bar:
                    entry_price = opens[i]
                    if not np.isfinite(entry_price) or entry_price <= 0.0:
                        side = 0
                        continue
                gain = side * (closes[i] / entry_price - 1.0)
                target_hit = np.isfinite(gain) and gain >= params.profit_target
                time_hit = i - entry_bar + 1 >= params.max_hold
                if target_hit or time_hit:
                    side = 0
                decisions[i] = side
            elif entries[i] in (-1, 1) and np.isfinite(closes[i]) and closes[i] > 0.0:
                side = int(entries[i])
                entry_bar = i + 1
                entry_price = np.nan
                decisions[i] = side
        df["signal"] = decisions
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = 1.0
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check():
    strategy = GeneratedStrategy()
    params = TopologyParams(graph_window=6, percentile_window=20, max_hold=2, profit_target=0.02)
    index = pd.date_range("2020-01-01", periods=10, freq="B")
    data = pd.DataFrame({
        "open": 100.0, "high": 104.0, "low": 99.0,
        "close": [100.0, 103.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0],
        "volume": 1000.0,
    }, index=index)
    entries = pd.DataFrame({"entry_direction": [1, 0, 0, 0, 0, -1, 0, 0, 0, 0]}, index=index)
    ctx = StrategyContext(symbol="SPY", timeframe="1d", warmup_bars=strategy.warmup_bars(params))
    signals = strategy.generate_signals(data, entries, ctx, params).data
    assert signals["signal"].tolist() == [0, 1, 0, 0, 0, 0, -1, -1, 0, 0]
    assert (signals["size"] > 0).all()

    rng = np.random.default_rng(7)
    close = 100.0 * np.exp(np.cumsum(rng.normal(0.0002, 0.01, 240)))
    index = pd.date_range("2021-01-01", periods=len(close), freq="B")
    history = pd.DataFrame({
        "open": close, "high": close * 1.01, "low": close * 0.99,
        "close": close, "volume": 1000.0,
    }, index=index)
    full = strategy.indicators(history, params)
    prefix = strategy.indicators(history.iloc[:160], params)
    pd.testing.assert_frame_equal(full.iloc[:160], prefix)
    assert full["triangle_closure"].dropna().between(0.0, 1.0).all()
    whole_signals = strategy.generate_signals(history, full, ctx, params).data
    prefix_signals = strategy.generate_signals(history.iloc[:160], prefix, ctx, params).data
    pd.testing.assert_frame_equal(whole_signals.iloc[:160], prefix_signals)
    assert set(whole_signals["signal"].unique()) <= {-1, 0, 1}


if __name__ == "__main__":
    _self_check()
