from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class Params:
    fd_window: int = 24
    frac_order: float = 0.45
    norm_window: int = 32
    drawdown_window: int = 20
    atr_window: int = 14
    memory_window: int = 5
    depth_atr: float = 1.5
    excursion: float = 1.25
    reclaim_level: float = -0.35
    trail_atr: float = 0.8
    hold_bars: int = 2


class GeneratedStrategy(BaseStrategy[Params]):
    strategy_id = "gen_local_1791244572"

    @classmethod
    def params_type(cls) -> type[Params]:
        return Params

    def warmup_bars(self, params: Params) -> int:
        return int(max(params.fd_window + params.norm_window + params.memory_window + 2,
                       params.drawdown_window, params.atr_window + 1))

    def indicators(self, data: pd.DataFrame, params: Params) -> pd.DataFrame:
        windows = (params.fd_window, params.norm_window, params.drawdown_window,
                   params.atr_window, params.memory_window)
        if any(w < 2 for w in windows):
            raise ValueError("Lookbacks must be at least two.")
        values = (params.frac_order, params.depth_atr, params.excursion,
                  params.reclaim_level, params.trail_atr)
        if not all(np.isfinite(v) for v in values):
            raise ValueError("Parameters must be finite.")
        if not 0 < params.frac_order < 1 or min(params.depth_atr, params.excursion, params.trail_atr) <= 0:
            raise ValueError("Invalid fractional order or thresholds.")
        if params.hold_bars not in (1, 2):
            raise ValueError("Holding horizon must be one or two sessions.")

        out = pd.DataFrame(index=data.index)
        close = data["close"].astype(float).replace([np.inf, -np.inf], np.nan)
        previous = close.shift(1)
        out["true_range"] = pd.concat([
            data["high"] - data["low"],
            (data["high"] - previous).abs(),
            (data["low"] - previous).abs(),
        ], axis=1).max(axis=1).replace([np.inf, -np.inf], np.nan)
        out["atr"] = out["true_range"].rolling(params.atr_window).mean().where(lambda x: x > 0)
        out["rolling_high"] = close.rolling(params.drawdown_window).max()
        out["depth"] = (out["rolling_high"] - close) / out["atr"]

        # Finite, causal binomial expansion of (1 - lag)^d.
        weights = np.ones(params.fd_window, dtype=float)
        for k in range(1, params.fd_window):
            weights[k] = weights[k - 1] * (k - 1 - params.frac_order) / k
        fractional = close * weights[0]
        for k in range(1, params.fd_window):
            fractional = fractional + weights[k] * close.shift(k)
        out["fractional_close"] = fractional
        out["fd_mean"] = fractional.rolling(params.norm_window).mean().shift(1)
        out["fd_std"] = fractional.rolling(params.norm_window).std(ddof=0).shift(1)
        out["fd_z"] = (fractional - out["fd_mean"]) / out["fd_std"].where(out["fd_std"] > 0)
        out["z_delta"] = out["fd_z"].diff()
        prior = out["fd_z"].shift(1)
        out["recent_trough"] = prior.rolling(params.memory_window).min()
        out["recent_crest"] = prior.rolling(params.memory_window).max()
        deep = out["depth"] >= params.depth_atr
        reclaim = (prior <= params.reclaim_level) & (out["fd_z"] > params.reclaim_level)
        failure = (prior >= params.reclaim_level) & (out["fd_z"] < params.reclaim_level)
        long_entry = deep & reclaim & (out["recent_trough"] <= -params.excursion)
        short_entry = deep & failure & (out["recent_crest"] >= params.excursion)
        out["candidate"] = np.select([long_entry, short_entry], [1, -1], default=0).astype(int)
        out["strength"] = (out["z_delta"].abs() / params.excursion).clip(0, 1).fillna(0)
        return out.replace([np.inf, -np.inf], np.nan)

    def generate_signals(
        self, data: pd.DataFrame, indicators: pd.DataFrame,
        ctx: StrategyContext, params: Params,
    ) -> SignalFrame:
        df = pd.DataFrame({"signal": 0, "size": 0.25}, index=data.index)
        signals = np.zeros(len(data), dtype=int)
        sizes = np.full(len(data), 0.25, dtype=float)
        position, decision_bar = 0, -1
        water, stop, entry_size = np.nan, np.nan, 0.25
        rows = zip(data["close"].to_numpy(), indicators["atr"].to_numpy(),
                   indicators["candidate"].to_numpy(), indicators["strength"].to_numpy(),
                   indicators["z_delta"].to_numpy())
        for i, (close, atr, candidate, strength, delta) in enumerate(rows):
            if not np.isfinite([close, atr, strength, delta]).all() or close <= 0 or atr <= 0:
                position = 0
                water, stop = np.nan, np.nan
                continue
            if position:
                # The first in-trade close occurs one bar after the entry decision.
                if position == 1:
                    water = close if np.isnan(water) else max(water, close)
                    proposed = water - params.trail_atr * atr
                    stop = proposed if np.isnan(stop) else max(stop, proposed)
                    stopped = close <= stop
                else:
                    # Mirror the long high-water trail for short positions.
                    water = close if np.isnan(water) else min(water, close)
                    proposed = water + params.trail_atr * atr
                    stop = proposed if np.isnan(stop) else min(stop, proposed)
                    stopped = close >= stop
                if stopped or i - decision_bar >= params.hold_bars or position * delta < 0:
                    position = 0
                    water, stop = np.nan, np.nan
                    continue
            elif candidate in (-1, 1):
                position, decision_bar = int(candidate), i
                water, stop = np.nan, np.nan
                entry_size = float(0.25 + 0.70 * np.clip(strength, 0, 1))
            signals[i] = position
            if position:
                sizes[i] = entry_size

        df["signal"] = signals
        df["size"] = sizes
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = df["size"].shift(1).fillna(0.25).astype(float)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check() -> None:
    index = pd.date_range("2020-01-01", periods=7)
    close = pd.Series([100., 102., 99., 100., 100., 103., 104.], index=index)
    data = pd.DataFrame({"open": close, "high": close + 1, "low": close - 1,
                         "close": close, "volume": 1000.}, index=index)
    derived = pd.DataFrame({"atr": 1., "candidate": [1, 0, 0, -1, 0, 0, 0],
                            "strength": [0.2, 0., 0., 0.8, 0., 0., 0.],
                            "z_delta": [1., 1., 1., -1., -1., -1., 0.]}, index=index)
    strategy = object.__new__(GeneratedStrategy)
    result = strategy.generate_signals(data, derived, None, Params()).data
    assert result["signal"].tolist() == [0, 1, 1, 0, -1, -1, 0]
    assert np.allclose(result["size"].iloc[[1, 2, 4, 5]], [0.39, 0.39, 0.81, 0.81])
    assert (result["size"] > 0).all()
    early = derived.copy()
    early.loc[index[1], "z_delta"] = -1.
    assert strategy.generate_signals(data, early, None, Params()).data["signal"].iloc[2] == 0
    empty = strategy.indicators(data, Params())
    assert empty["candidate"].eq(0).all()


if __name__ == "__main__":
    _self_check()
