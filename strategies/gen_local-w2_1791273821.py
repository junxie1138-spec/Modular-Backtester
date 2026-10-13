from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class Params:
    frac_order: float = 0.45
    frac_terms: int = 32
    position_window: int = 20
    edge: float = 0.2
    pressure_window: int = 3
    atr_window: int = 14
    stop_k: float = 1.5
    hold_bars: int = 4

    def __post_init__(self):
        if not (0.0 < self.frac_order < 1.0 and 0.0 < self.edge < 0.5):
            raise ValueError("Invalid fractional order or edge")
        if min(self.frac_terms, self.position_window, self.pressure_window, self.atr_window) < 2:
            raise ValueError("Lookbacks must be at least two")
        if not (self.stop_k > 0.0 and 3 <= self.hold_bars <= 5):
            raise ValueError("Invalid stop or holding horizon")


class GeneratedStrategy(BaseStrategy[Params]):
    strategy_id = "gen_local-w2_1791273821"

    @classmethod
    def params_type(cls):
        return Params

    def warmup_bars(self, params):
        return int(max(params.frac_terms + params.position_window - 1,
                       params.pressure_window + 1, params.atr_window + 1))

    def indicators(self, data, params):
        out = pd.DataFrame(index=data.index)
        close = data["close"].astype(float)
        # ponytail: finite fractional memory; increase frac_terms for longer tails.
        fractional = close.copy()
        weight = 1.0
        for lag in range(1, params.frac_terms):
            weight *= (lag - 1 - params.frac_order) / lag
            fractional = fractional + weight * close.shift(lag)
        out["fractional_close"] = fractional
        out["fractional_low"] = fractional.rolling(params.position_window).min()
        out["fractional_high"] = fractional.rolling(params.position_window).max()
        span = (out["fractional_high"] - out["fractional_low"]).replace(0.0, np.nan)
        out["relative_position"] = (fractional - out["fractional_low"]) / span
        out["bar_range"] = data["high"] - data["low"]
        out["settlement_location"] = (
            (2.0 * close - data["high"] - data["low"])
            / out["bar_range"].replace(0.0, np.nan)
        ).where(out["bar_range"].ne(0.0), 0.0)
        volume = data["volume"].astype(float).where(data["volume"] >= 0)
        out["signed_volume"] = out["settlement_location"] * volume
        out["volume_sum"] = volume.rolling(params.pressure_window).sum()
        out["pressure"] = (
            out["signed_volume"].rolling(params.pressure_window).sum()
            / out["volume_sum"].replace(0.0, np.nan)
        )
        out["previous_pressure"] = out["pressure"].shift(1)
        previous_close = close.shift(1)
        out["true_range"] = pd.concat(
            [out["bar_range"], (data["high"] - previous_close).abs(),
             (data["low"] - previous_close).abs()], axis=1
        ).max(axis=1, skipna=False)
        out["atr"] = out["true_range"].rolling(params.atr_window).mean()
        long_entry = (
            (out["relative_position"] <= params.edge)
            & (out["pressure"] > 0.0) & (out["previous_pressure"] <= 0.0)
        )
        short_entry = (
            (out["relative_position"] >= 1.0 - params.edge)
            & (out["pressure"] < 0.0) & (out["previous_pressure"] >= 0.0)
        )
        out["entry_direction"] = np.select([long_entry, short_entry], [1, -1], default=0)
        return out.replace([np.inf, -np.inf], np.nan)

    def generate_signals(self, data, indicators, ctx, params):
        close = data["close"].to_numpy(dtype=float)
        opens = data["open"].to_numpy(dtype=float)
        atr = indicators["atr"].to_numpy(dtype=float)
        votes = indicators["entry_direction"].fillna(0).to_numpy(dtype=int)
        decisions = np.zeros(len(data), dtype=int)
        position = 0
        entry_bar = -1
        stop = np.nan
        stop_distance = 0.0
        for i in range(len(data)):
            if position:
                if i == entry_bar:
                    # Entry fills at this bar's open; ATR was frozen at the prior close.
                    stop = opens[i] - position * stop_distance
                stopped = (
                    not np.isfinite(close[i]) or not np.isfinite(stop)
                    or position * (close[i] - stop) < 0.0
                )
                expired = i - entry_bar + 1 >= params.hold_bars
                if stopped or expired:
                    position = 0
            elif votes[i] and np.isfinite(close[i]) and np.isfinite(atr[i]) and atr[i] > 0.0:
                position = int(votes[i])
                entry_bar = i + 1
                stop_distance = params.stop_k * atr[i]
            decisions[i] = position
        df = pd.DataFrame({"signal": decisions, "size": 0.95}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check():
    params = Params()
    index = pd.date_range("2020-01-01", periods=9)
    data = pd.DataFrame({
        "open": 100.0, "high": 112.0, "low": 94.0,
        "close": [100.0, 100.0, 110.0, 96.0, 100.0, 100.0, 100.0, 104.0, 100.0],
        "volume": 1000.0
    }, index=index)
    indicators = pd.DataFrame({
        "atr": 2.0, "entry_direction": [1, 0, 0, 0, 0, -1, 0, 0, 0]
    }, index=index)
    result = GeneratedStrategy.generate_signals(None, data, indicators, None, params).data
    assert result["signal"].tolist() == [0, 1, 1, 1, 0, 0, -1, -1, 0]
    data["close"] = 100.0
    indicators["entry_direction"] = [1, 0, 0, 0, 0, 0, 0, 0, 0]
    result = GeneratedStrategy.generate_signals(None, data, indicators, None, params).data
    assert result["signal"].tolist() == [0, 1, 1, 1, 1, 0, 0, 0, 0]
    data = pd.concat([data] * 10, ignore_index=True)
    data["close"] = 100.0 + 2.0 * np.sin(np.arange(len(data)) / 3.0)
    full = GeneratedStrategy.indicators(None, data, params)
    prefix = GeneratedStrategy.indicators(None, data.iloc[:65], params)
    pd.testing.assert_frame_equal(full.iloc[:65], prefix)
    assert full["relative_position"].iloc[:50].isna().all()


if __name__ == "__main__":
    _self_check()
