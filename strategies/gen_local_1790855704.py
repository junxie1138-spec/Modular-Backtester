from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class DrawdownParams:
    peak_window: int = 20
    corr_window: int = 40
    atr_window: int = 14
    depth_atr: float = 1.0
    edge_atr: float = 0.10
    corr_cutoff: float = -0.10
    breakeven_pct: float = 0.004
    trail_atr: float = 1.25
    min_size: float = 0.25
    max_size: float = 0.95

    def __post_init__(self):
        if min(self.peak_window, self.corr_window, self.atr_window) < 2:
            raise ValueError("Windows must be at least two bars.")
        if not (self.depth_atr > 0 and self.edge_atr > 0 and self.trail_atr > 0):
            raise ValueError("Depth, edge and trail must be positive.")
        if not (-1 < self.corr_cutoff < 0 and self.breakeven_pct > 0):
            raise ValueError("Invalid correlation cutoff or breakeven trigger.")
        if not (0 < self.min_size <= self.max_size <= 1):
            raise ValueError("Sizes must satisfy 0 < min_size <= max_size <= 1.")


class GeneratedStrategy(BaseStrategy[DrawdownParams]):
    strategy_id = "gen_local_1790855704"

    @classmethod
    def params_type(cls):
        return DrawdownParams

    def warmup_bars(self, params):
        return int(max(params.peak_window + params.corr_window + 2, params.atr_window + 1))

    def indicators(self, data, params):
        p = params
        out = pd.DataFrame(index=data.index)
        close = data["close"].where(data["close"] > 0)
        out["peak"] = close.rolling(p.peak_window, min_periods=p.peak_window).max()
        out["drawdown"] = 1.0 - close / out["peak"]
        out["depth_change"] = out["drawdown"].diff()
        out["lag_change"] = out["depth_change"].shift(1)
        out["lag_variance"] = out["lag_change"].rolling(p.corr_window).var()
        out["covariance"] = out["depth_change"].rolling(p.corr_window).cov(out["lag_change"])
        out["autocorrelation"] = out["depth_change"].rolling(p.corr_window).corr(out["lag_change"])
        out["beta"] = out["covariance"] / out["lag_variance"].where(out["lag_variance"] > 1e-12)
        previous_close = close.shift(1)
        out["true_range"] = pd.concat([
            data["high"] - data["low"],
            (data["high"] - previous_close).abs(),
            (data["low"] - previous_close).abs(),
        ], axis=1).max(axis=1)
        out["atr"] = out["true_range"].rolling(p.atr_window).mean()
        out["atr_fraction"] = (out["atr"] / close).where(out["atr"] > 0)
        out["depth_atr"] = out["drawdown"] / out["atr_fraction"]
        out["edge_atr"] = -out["beta"] * out["depth_change"] / out["atr_fraction"]
        out = out.replace([np.inf, -np.inf], np.nan)
        out["eligible"] = (
            (out["depth_atr"] >= p.depth_atr)
            & (out["depth_change"] > 0)
            & (out["autocorrelation"] <= p.corr_cutoff)
            & (out["edge_atr"] >= p.edge_atr)
        ).fillna(False)
        out["strength"] = (out["edge_atr"] / (2.0 * p.edge_atr)).clip(0.0, 1.0).fillna(0.0)
        out["entry_size"] = p.min_size + (p.max_size - p.min_size) * out["strength"]
        return out

    def generate_signals(self, data, indicators, ctx, params):
        p = params
        n = len(data)
        signals = np.zeros(n, dtype=int)
        sizes = np.full(n, p.min_size, dtype=float)
        eligible = indicators["eligible"].fillna(False).to_numpy(dtype=bool)
        atr = indicators["atr"].to_numpy(dtype=float)
        entry_sizes = indicators["entry_size"].fillna(p.min_size).to_numpy(dtype=float)
        opens = data["open"].to_numpy(dtype=float)
        highs = data["high"].to_numpy(dtype=float)
        lows = data["low"].to_numpy(dtype=float)
        closes = data["close"].to_numpy(dtype=float)
        active = False
        armed = False
        entry = stop = peak = held_size = 0.0
        entry_bar = -1

        for i in range(n):
            if not active and i > 0 and signals[i - 1] == 1:
                # The previous close's decision fills at this bar's open.
                entry = opens[i]
                entry_bar = i
                held_size = sizes[i - 1]
                peak = entry
                stop = entry - p.trail_atr * atr[i - 1]
                active = True
                armed = False

            if active:
                # Existing stop is checked before this bar can raise it.
                exit_now = lows[i] <= stop
                if not exit_now:
                    peak = max(peak, highs[i])
                    armed = armed or peak >= entry * (1.0 + p.breakeven_pct)
                    if armed:
                        stop = max(stop, entry)
                        if np.isfinite(atr[i]) and atr[i] > 0:
                            stop = max(stop, peak - p.trail_atr * atr[i])
                    exit_now = closes[i] <= stop or i - entry_bar + 1 >= 2
                sizes[i] = held_size
                if exit_now:
                    active = False
                    # No re-entry on an exit decision bar.
                    continue
                signals[i] = 1
            elif eligible[i]:
                signals[i] = 1
                sizes[i] = entry_sizes[i]

        df = pd.DataFrame({"signal": signals, "size": sizes}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = df["size"].shift(1).fillna(p.min_size).astype(float)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check():
    strategy = GeneratedStrategy()
    p = DrawdownParams()
    index = pd.date_range("2020-01-01", periods=5)
    data = pd.DataFrame({
        "open": [100.0] * 5,
        "high": [100.0, 101.0, 100.5, 100.0, 100.0],
        "low": [99.9, 99.9, 99.8, 99.9, 99.9],
        "close": [100.0, 100.8, 99.9, 100.0, 100.0],
        "volume": [1000.0] * 5,
    }, index=index)
    ind = pd.DataFrame({
        "eligible": [True, False, False, False, False],
        "atr": [1.0] * 5,
        "entry_size": [0.70] * 5,
    }, index=index)
    result = strategy.generate_signals(data, ind, None, p).data
    assert result["signal"].tolist() == [0, 1, 1, 0, 0]
    assert result.loc[index[1], "size"] == 0.70
    trailing = data.copy()
    trailing.loc[index[1], "high"] = 102.5
    assert strategy.generate_signals(trailing, ind, None, p).data["signal"].tolist() == [0, 1, 0, 0, 0]
    breakeven = data.copy()
    breakeven.loc[index[1], "close"] = 99.95
    ind.loc[index[1], "atr"] = 20.0
    assert strategy.generate_signals(breakeven, ind, None, p).data["signal"].tolist() == [0, 1, 0, 0, 0]
    data.loc[index[1], "low"] = 98.0
    assert strategy.generate_signals(data, ind, None, p).data["signal"].tolist() == [0, 1, 0, 0, 0]
    flat = data.copy()
    flat[["open", "high", "low", "close"]] = 100.0
    flat_ind = strategy.indicators(flat, p)
    assert not flat_ind["eligible"].any()
    assert (strategy.generate_signals(flat, flat_ind, None, p).data["signal"] == 0).all()


if __name__ == "__main__":
    _self_check()
