from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class StrategyParams:
    compression_bars: int = 5
    baseline_bars: int = 20
    atr_bars: int = 14
    compression_ratio: float = 0.75
    gap_atr: float = 0.35
    stop_atr: float = 0.8
    hold_bars: int = 2
    min_size: float = 0.25


class GeneratedStrategy(BaseStrategy[StrategyParams]):
    strategy_id = "gen_local-w2_1790972287"

    @classmethod
    def params_type(cls):
        return StrategyParams

    def warmup_bars(self, params):
        return int(max(params.compression_bars, params.baseline_bars, params.atr_bars) + 1)

    def indicators(self, data, params):
        if min(params.compression_bars, params.baseline_bars, params.atr_bars) < 1:
            raise ValueError("Lookbacks must be positive.")
        if not (0 < params.compression_ratio < 1 and params.gap_atr > 0
                and params.stop_atr > 0 and 1 <= params.hold_bars <= 2
                and 0 < params.min_size <= 1):
            raise ValueError("Invalid strategy parameters.")
        out = pd.DataFrame(index=data.index)
        out["prior_close"] = data["close"].shift(1)
        out["true_range"] = pd.concat([
            data["high"] - data["low"],
            (data["high"] - out["prior_close"]).abs(),
            (data["low"] - out["prior_close"]).abs(),
        ], axis=1).max(axis=1)
        out["atr"] = out["true_range"].rolling(params.atr_bars).mean()
        out["prior_atr"] = out["atr"].shift(1).replace(0, np.nan)
        out["short_range"] = out["true_range"].rolling(params.compression_bars).mean().shift(1)
        out["baseline_range"] = out["true_range"].rolling(params.baseline_bars).mean().shift(1)
        out["compression"] = out["short_range"] / out["baseline_range"].replace(0, np.nan)
        out["floor"] = data["low"].rolling(params.compression_bars).min().shift(1)
        out["ceiling"] = data["high"].rolling(params.compression_bars).max().shift(1)
        out["envelope_atr"] = (out["ceiling"] - out["floor"]) / out["prior_atr"]
        out["gap_down"] = (out["prior_close"] - data["open"]) / out["prior_atr"]
        out["recovery"] = ((data["close"] - data["open"]) /
                           (data["high"] - data["low"]).replace(0, np.nan))
        out["conviction"] = (
            out["gap_down"].clip(0, 1)
            * (1 - out["compression"]).clip(0, 1)
            * out["recovery"].clip(0, 1)
        ).fillna(0).clip(0, 1)
        out["entry"] = (
            (out["compression"] <= params.compression_ratio)
            & (out["envelope_atr"] <= 2.5)
            & (out["gap_down"] >= params.gap_atr)
            & (data["open"] < out["floor"])
            & (data["close"] > out["floor"])
            & (data["close"] < out["prior_close"])
            & (out["recovery"] >= 0.45)
        ).fillna(False)
        return out.replace([np.inf, -np.inf], np.nan)

    def generate_signals(self, data, indicators, ctx, params):
        df = pd.DataFrame(index=data.index)
        signals = np.zeros(len(data), dtype=int)
        sizes = np.full(len(data), params.min_size, dtype=float)
        active = False
        age = 0
        high_water = -np.inf
        stop = -np.inf
        position_size = params.min_size
        closes = data["close"].to_numpy(dtype=float)
        atr = indicators["atr"].to_numpy(dtype=float)
        entries = indicators["entry"].fillna(False).to_numpy(dtype=bool)
        conviction = indicators["conviction"].fillna(0).to_numpy(dtype=float)
        for i in range(len(data)):
            if not (np.isfinite(closes[i]) and np.isfinite(atr[i]) and atr[i] > 0):
                active = False
                continue
            if active:
                # The first iteration after an entry decision is the fill session.
                age += 1
                high_water = max(high_water, closes[i])
                stop = max(stop, high_water - params.stop_atr * atr[i])
                if closes[i] <= stop or age >= params.hold_bars:
                    active = False
            elif entries[i]:
                active = True
                age = 0
                high_water = -np.inf
                stop = -np.inf
                position_size = params.min_size + (1 - params.min_size) * conviction[i]
            if active:
                signals[i] = 1
                sizes[i] = position_size
        df["signal"] = signals
        df["size"] = sizes
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = df["size"].shift(1).fillna(params.min_size).astype(float)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


if __name__ == "__main__":
    strategy = GeneratedStrategy()
    params = StrategyParams()
    index = pd.date_range("2020-01-01", periods=7)
    data = pd.DataFrame({
        "open": [100.0] * 7, "high": [101.0] * 7,
        "low": [98.0] * 7,
        "close": [100.0, 100.0, 100.0, 100.0, 100.0, 99.0, 100.0],
        "volume": [1000.0] * 7,
    }, index=index)
    ind = pd.DataFrame({
        "atr": [1.0] * 7,
        "entry": [True, False, False, True, False, False, False],
        "conviction": [0.2, 0, 0, 0.8, 0, 0, 0],
    }, index=index)
    ctx = StrategyContext("SPY", "1d", strategy.warmup_bars(params))
    result = strategy.generate_signals(data, ind, ctx, params).data
    assert result["signal"].tolist() == [0, 1, 1, 0, 1, 1, 0]
    assert np.isclose(result["size"].iloc[1], 0.4)
    assert np.isclose(result["size"].iloc[4], 0.85)
    assert result["signal"].isin([0, 1]).all() and (result["size"] > 0).all()
    params.hold_bars = 1
    result = strategy.generate_signals(data, ind, ctx, params).data
    assert result["signal"].tolist() == [0, 1, 0, 0, 1, 0, 0]
    warm = strategy.indicators(data, params)
    assert not warm["entry"].any()
    assert strategy.warmup_bars(params) == 21
