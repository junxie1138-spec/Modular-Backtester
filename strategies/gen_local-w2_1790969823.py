from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class VisibilityParams:
    graph_window: int = 24
    mean_window: int = 24
    atr_window: int = 14
    topology_threshold: float = 0.5
    min_dislocation: float = 1.2
    breakeven_pct: float = 0.025
    atr_mult: float = 2.8
    hold_bars: int = 18

    def __post_init__(self):
        if min(self.graph_window, self.mean_window, self.atr_window) < 3:
            raise ValueError("Lookbacks must be at least three.")
        if self.hold_bars < 1 or not 0 < self.topology_threshold < 1:
            raise ValueError("Invalid holding period or topology threshold.")
        if not all(np.isfinite(v) and v > 0 for v in (
            self.min_dislocation, self.breakeven_pct, self.atr_mult
        )):
            raise ValueError("Risk and dislocation parameters must be positive.")


class GeneratedStrategy(BaseStrategy[VisibilityParams]):
    strategy_id = "gen_local-w2_1790969823"

    @classmethod
    def params_type(cls):
        return VisibilityParams

    def warmup_bars(self, params: VisibilityParams) -> int:
        return int(max(params.graph_window, params.mean_window, params.atr_window + 1))

    def indicators(self, data: pd.DataFrame, params: VisibilityParams) -> pd.DataFrame:
        out = pd.DataFrame(index=data.index)
        close = data["close"].astype(float)
        log_price = np.log(close.where(close > 0)).to_numpy()
        n, window = len(data), params.graph_window
        # Natural visibility: every intervening point lies below the connecting line.
        # ponytail: O(n * window**2) triangle count; sparse adjacency if windows become large.
        for label, direction in (("upper", 1), ("lower", -1)):
            x = direction * log_price
            edges = np.zeros((n, window), dtype=bool)
            best = np.full(n, -np.inf)
            for lag in range(1, min(window, n)):
                slope = np.full(n, np.nan)
                slope[lag:] = (x[:-lag] - x[lag:]) / lag
                valid = np.isfinite(slope)
                edges[:, lag] = valid & (slope > best)
                best = np.maximum(best, np.where(valid, slope, -np.inf))
            degree = edges.sum(axis=1)
            triangles = np.zeros(n, dtype=float)
            for near in range(1, min(window - 1, n)):
                for far in range(near + 1, min(window, n)):
                    triangles[near:] += (
                        edges[near:, near] & edges[near:, far]
                        & edges[:-near, far - near]
                    )
            out[label + "_degree"] = degree
            out[label + "_triangles"] = triangles
        ready = close.where(np.isfinite(close) & (close > 0)).rolling(window).count().eq(window)
        out["topology"] = (
            (out["upper_triangles"] - out["lower_triangles"])
            / (out["upper_triangles"] + out["lower_triangles"] + 1.0)
        ).where(ready)
        previous = close.shift(1)
        out["true_range"] = pd.concat([
            data["high"] - data["low"],
            (data["high"] - previous).abs(),
            (data["low"] - previous).abs(),
        ], axis=1).max(axis=1).where(previous.notna())
        out["atr"] = out["true_range"].rolling(params.atr_window).mean()
        out["mean"] = close.rolling(params.mean_window).mean()
        out["dislocation"] = (close - out["mean"]) / out["atr"].where(out["atr"] > 0)
        out["turn"] = close.diff()
        long_entry = (
            (out["topology"] > params.topology_threshold)
            & (out["upper_degree"] >= 4)
            & (out["dislocation"] < -params.min_dislocation)
            & (out["turn"] > 0)
        )
        short_entry = (
            (out["topology"] < -params.topology_threshold)
            & (out["lower_degree"] >= 4)
            & (out["dislocation"] > params.min_dislocation)
            & (out["turn"] < 0)
        )
        out["entry"] = np.select([long_entry, short_entry], [1, -1], default=0).astype(int)
        out["strength"] = (
            0.35 + 0.65 * out["topology"].abs()
            * (out["dislocation"].abs() / (2 * params.min_dislocation)).clip(0, 1)
        ).replace([np.inf, -np.inf], np.nan).fillna(0.35).clip(0.35, 1.0)
        return out

    def generate_signals(
        self, data: pd.DataFrame, indicators: pd.DataFrame,
        ctx: StrategyContext, params: VisibilityParams,
    ) -> SignalFrame:
        n = len(data)
        signals = np.zeros(n, dtype=int)
        sizes = np.full(n, 0.35, dtype=float)
        opens = data["open"].to_numpy(dtype=float)
        closes = data["close"].to_numpy(dtype=float)
        atr = indicators["atr"].to_numpy(dtype=float)
        entries = indicators["entry"].fillna(0).to_numpy(dtype=int)
        strength = indicators["strength"].to_numpy(dtype=float)
        side, entry_bar, armed = 0, 0, False
        entry_price, stop, trade_size = 0.0, 0.0, 0.35
        for i in range(n):
            # A decision from the previous close enters at this bar's open.
            if side == 0 and i > 0 and signals[i - 1] != 0:
                if np.isfinite(opens[i]) and opens[i] > 0:
                    side = int(signals[i - 1])
                    entry_price, entry_bar = opens[i], i
                    trade_size = sizes[i - 1]
                    stop = side * entry_price - params.atr_mult * atr[i - 1]
                    armed = False
            if side:
                if not np.isfinite(closes[i]) or closes[i] <= 0 or not np.isfinite(atr[i]) or atr[i] <= 0:
                    side = 0
                    continue
                # Signed-price stops only rise: long stops rise, short buy-stops fall.
                signed_close = side * closes[i]
                if side * (closes[i] / entry_price - 1.0) >= params.breakeven_pct:
                    armed = True
                if armed:
                    stop = max(stop, side * entry_price, signed_close - params.atr_mult * atr[i])
                if signed_close <= stop or i - entry_bar + 1 >= params.hold_bars:
                    side = 0
                    continue
                signals[i], sizes[i] = side, trade_size
            elif i >= self.warmup_bars(params) - 1:
                if np.isfinite(atr[i]) and atr[i] > 0 and np.isfinite(strength[i]):
                    signals[i] = entries[i]
                    sizes[i] = float(np.clip(strength[i], 0.35, 1.0))
        df = pd.DataFrame({"signal": signals, "size": sizes}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = df["size"].shift(1).fillna(0.35).astype(float)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check():
    strategy = GeneratedStrategy()
    params = VisibilityParams(graph_window=3, mean_window=3, atr_window=3, atr_mult=2.0)
    index = pd.date_range("2020-01-01", periods=9)
    ctx = StrategyContext(symbol="SPY", timeframe="1d", warmup_bars=4)
    for direction in (1, -1):
        close = 100.0 + direction * np.array([0, 0, 0, 0, 3, 5, 2, 0, 0])
        data = pd.DataFrame({
            "open": 100.0, "high": close + 1, "low": close - 1,
            "close": close, "volume": 1000.0,
        }, index=index)
        ind = pd.DataFrame({
            "atr": 1.0, "entry": [0, 0, 0, direction, 0, 0, 0, 0, 0],
            "strength": 0.7,
        }, index=index)
        result = strategy.generate_signals(data, ind, ctx, params).data
        assert result["signal"].tolist() == [0, 0, 0, 0, direction, direction, direction, 0, 0]
        assert result.loc[index[4], "size"] == 0.7
        assert (result["size"] > 0).all()
        flat = data.assign(open=100.0, high=101.0, low=99.0, close=100.0)
        calculated = strategy.indicators(flat, params)
        assert calculated["entry"].eq(0).all()
        prefix = strategy.indicators(data.iloc[:6], params)
        pd.testing.assert_frame_equal(prefix, strategy.indicators(data, params).iloc[:6])


if __name__ == "__main__":
    _self_check()
