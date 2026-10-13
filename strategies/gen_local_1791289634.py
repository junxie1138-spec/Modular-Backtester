from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class Params:
    graph_window: int = 16
    conductance_cutoff: float = 0.20
    atr_window: int = 14
    momentum_window: int = 5
    trail_atr: float = 2.5
    hold_bars: int = 8

    def __post_init__(self):
        if self.graph_window < 4 or self.graph_window % 2:
            raise ValueError("graph_window must be even and at least four")
        if self.atr_window < 1 or self.momentum_window < 1:
            raise ValueError("indicator windows must be positive")
        if not 0.0 < self.conductance_cutoff < 1.0:
            raise ValueError("conductance_cutoff must lie between zero and one")
        if not np.isfinite(self.trail_atr) or self.trail_atr <= 0:
            raise ValueError("trail_atr must be finite and positive")
        if not 5 <= self.hold_bars <= 10:
            raise ValueError("hold_bars must be between five and ten")


class GeneratedStrategy(BaseStrategy[Params]):
    strategy_id = "gen_local_1791289634"

    @classmethod
    def params_type(cls):
        return Params

    def warmup_bars(self, params):
        return int(max(200, params.graph_window + 1,
                       params.atr_window + 1, params.momentum_window + 1, 3))

    def indicators(self, data, params):
        out = pd.DataFrame(index=data.index)
        close = data["close"].astype(float)
        out["previous_close"] = close.shift(1)
        out["true_range"] = pd.concat([
            data["high"] - data["low"],
            (data["high"] - out["previous_close"]).abs(),
            (data["low"] - out["previous_close"]).abs(),
        ], axis=1).max(axis=1, skipna=False)
        out["range_fraction"] = (
            out["true_range"] / out["previous_close"].where(out["previous_close"] > 0)
        ).replace([np.inf, -np.inf], np.nan)
        out["atr"] = out["true_range"].rolling(params.atr_window).mean()
        out["ma200"] = close.rolling(200).mean()
        out["momentum"] = close.pct_change(params.momentum_window, fill_method=None)
        out["pullback"] = close.pct_change(2, fill_method=None)
        out["conductance"] = np.nan

        w = params.graph_window
        if len(data) >= w:
            windows = np.lib.stride_tricks.sliding_window_view(
                out["range_fraction"].to_numpy(), w
            )
            valid = np.isfinite(windows).all(axis=1)
            n = len(windows)
            half = w // 2
            # Start with the mandatory adjacent-node edges of the visibility graph.
            left_volume = np.full(n, 2 * half - 1, dtype=float)
            right_volume = np.full(n, 2 * (w - half) - 1, dtype=float)
            cut = np.ones(n, dtype=float)
            # ponytail: O(N*w^2), keep windows small; use compiled edges for large w.
            for i in range(w - 2):
                interior_max = windows[:, i + 1].copy()
                for j in range(i + 2, w):
                    edge = (windows[:, i] > interior_max) & (
                        windows[:, j] > interior_max
                    )
                    if i < half and j < half:
                        left_volume += 2 * edge
                    elif i >= half:
                        right_volume += 2 * edge
                    else:
                        cut += edge
                        left_volume += edge
                        right_volume += edge
                    interior_max = np.maximum(interior_max, windows[:, j])
            conductance = cut / np.minimum(left_volume, right_volume)
            conductance[~valid] = np.nan
            out.iloc[w - 1:, out.columns.get_loc("conductance")] = conductance

        ready = np.isfinite(out[[
            "atr", "ma200", "momentum", "pullback", "conductance"
        ]]).all(axis=1) & (out["atr"] > 0) & np.isfinite(close) & (close > 0)
        out["graph_regime"] = np.where(
            ready, (out["conductance"] >= params.conductance_cutoff).astype(int), -1
        )
        # Connected volatility episodes favor fading a two-day countertrend move.
        # A bottleneck between episodes favors five-day trend continuation.
        direction = pd.Series(np.where(
            out["graph_regime"] == 1,
            -np.sign(out["pullback"]),
            np.sign(out["momentum"]),
        ), index=data.index)
        out["candidate"] = np.select([
            ready & (close > out["ma200"]) & (direction > 0),
            ready & (close < out["ma200"]) & (direction < 0),
        ], [1, -1], default=0).astype(int)
        return out

    def generate_signals(self, data, indicators, ctx: StrategyContext, params):
        closes = data["close"].to_numpy(dtype=float)
        atrs = indicators["atr"].to_numpy(dtype=float)
        averages = indicators["ma200"].to_numpy(dtype=float)
        candidates = indicators["candidate"].to_numpy(dtype=int)
        signals = np.zeros(len(data), dtype=int)
        position = 0
        age = 0
        high_water = low_water = np.nan
        floor = -np.inf
        ceiling = np.inf

        for i in range(len(data)):
            close, atr, average = closes[i], atrs[i], averages[i]
            valid = (np.isfinite(close) and close > 0 and
                     np.isfinite(atr) and atr > 0 and np.isfinite(average))
            if not valid:
                position = 0
                age = 0
                high_water = low_water = np.nan
                floor, ceiling = -np.inf, np.inf
                continue

            if position:
                # The previous decision filled at today's open. Start tracking
                # closes on this first filled bar, not on the decision bar.
                age += 1
                high_water = close if np.isnan(high_water) else max(high_water, close)
                low_water = close if np.isnan(low_water) else min(low_water, close)
                floor = max(floor, high_water - params.trail_atr * atr)
                # Shorts also have a mirrored protective rebound stop.
                if position < 0:
                    ceiling = min(ceiling, low_water + params.trail_atr * atr)
                regime_exit = ((position > 0 and close <= average) or
                               (position < 0 and close >= average))
                # The required high-water floor applies to both directions:
                # on a short, crossing it takes profits after a downward excursion.
                if (close < floor or (position < 0 and close > ceiling) or
                        regime_exit or age >= params.hold_bars):
                    position = 0
                    age = 0
                    high_water = low_water = np.nan
                    floor, ceiling = -np.inf, np.inf
            elif candidates[i]:
                position = int(candidates[i])
                age = 0
                high_water = low_water = np.nan
                floor, ceiling = -np.inf, np.inf
            signals[i] = position

        df = pd.DataFrame({"signal": signals, "size": 0.95}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check():
    strategy = GeneratedStrategy()
    params = Params()
    t = np.arange(260, dtype=float)
    close = 100.0 + 0.03 * t + np.sin(t / 3.0)
    data = pd.DataFrame({
        "open": close, "high": close + 1, "low": close - 1,
        "close": close, "volume": np.full(len(t), 1000.0),
    }, index=pd.date_range("2020-01-01", periods=len(t)))
    full = strategy.indicators(data, params)
    prefix = strategy.indicators(data.iloc[:230], params)
    pd.testing.assert_frame_equal(full.iloc[:230], prefix)
    full_signals = strategy.generate_signals(data, full, None, params).data
    prefix_signals = strategy.generate_signals(
        data.iloc[:230], prefix, None, params
    ).data
    pd.testing.assert_frame_equal(full_signals.iloc[:230], prefix_signals)
    assert strategy.warmup_bars(params) >= 200
    assert full_signals["signal"].isin([-1, 0, 1]).all()
    assert (full_signals["size"] > 0).all()
    assert strategy.generate_signals(
        data.iloc[:0], full.iloc[:0], None, params
    ).data.empty

    sample = data.iloc[:6].copy()
    sample["close"] = [100.0, 101.0, 105.0, 102.0, 102.0, 102.0]
    fixture = pd.DataFrame({
        "atr": [1.0, 1.0, 1.0, 10.0, 1.0, 1.0],
        "ma200": 90.0, "candidate": [1, 0, 0, 0, 0, 0],
    }, index=sample.index)
    result = strategy.generate_signals(
        sample, fixture, None, Params(trail_atr=2.0)
    ).data
    # ATR expansion cannot loosen the floor; exit decision at bar 3 fills bar 4.
    assert result["signal"].tolist() == [0, 1, 1, 1, 0, 0]


if __name__ == "__main__":
    _self_check()
