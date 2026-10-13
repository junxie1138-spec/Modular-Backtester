from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class PressureParams:
    range_bars: int = 20
    roc_bars: int = 4
    pressure_threshold: float = 0.02
    profit_target: float = 0.025
    max_hold_bars: int = 8

    def __post_init__(self):
        if self.range_bars < 2 or self.roc_bars < 1 or self.max_hold_bars < 1:
            raise ValueError("Lookbacks and holding period must be positive.")
        if not (np.isfinite(self.pressure_threshold) and self.pressure_threshold > 0):
            raise ValueError("Pressure threshold must be finite and positive.")
        if not (np.isfinite(self.profit_target) and self.profit_target > 0):
            raise ValueError("Profit target must be finite and positive.")


class GeneratedStrategy(BaseStrategy[PressureParams]):
    strategy_id = "gen_local_1791293718"

    @classmethod
    def params_type(cls):
        return PressureParams

    @staticmethod
    def warmup_bars(params):
        return int(max(200, params.range_bars + 1, params.roc_bars + 2))

    @staticmethod
    def indicators(data, params):
        out = pd.DataFrame(index=data.index)
        close = data["close"]
        out["ma200"] = close.rolling(200, min_periods=200).mean()
        out["range_high"] = data["high"].rolling(params.range_bars, min_periods=params.range_bars).max().shift(1)
        out["range_low"] = data["low"].rolling(params.range_bars, min_periods=params.range_bars).min().shift(1)
        out["range_width"] = (out["range_high"] - out["range_low"]).where(lambda s: s > 0)
        out["position"] = (close - out["range_low"]) / out["range_width"]
        out["roc"] = close.pct_change(params.roc_bars, fill_method=None)
        out["acceleration"] = out["roc"].diff()
        out["long_pressure"] = out["acceleration"] / (1.0 - out["position"]).clip(lower=0.1)
        out["short_pressure"] = -out["acceleration"] / out["position"].clip(lower=0.1)
        out = out.replace([np.inf, -np.inf], np.nan)
        out["previous_long_pressure"] = out["long_pressure"].shift(1)
        out["previous_short_pressure"] = out["short_pressure"].shift(1)
        out["ready"] = out.notna().all(axis=1)
        out["long_entry"] = (
            out["ready"] & (close > out["ma200"]) & (out["roc"] > 0)
            & out["position"].between(0.5, 0.9, inclusive="neither")
            & (out["long_pressure"] > params.pressure_threshold)
            & (out["previous_long_pressure"] <= params.pressure_threshold)
        )
        out["short_entry"] = (
            out["ready"] & (close < out["ma200"]) & (out["roc"] < 0)
            & out["position"].between(0.1, 0.5, inclusive="neither")
            & (out["short_pressure"] > params.pressure_threshold)
            & (out["previous_short_pressure"] <= params.pressure_threshold)
        )
        return out

    @staticmethod
    def generate_signals(data, indicators, ctx, params):
        signals = np.zeros(len(data), dtype=int)
        opens = data["open"].to_numpy(dtype=float)
        closes = data["close"].to_numpy(dtype=float)
        longs = indicators["long_entry"].fillna(False).to_numpy(dtype=bool)
        shorts = indicators["short_entry"].fillna(False).to_numpy(dtype=bool)
        direction = 0
        entry_price = np.nan
        entry_bar = -1
        for i in range(len(data)):
            if direction:
                if i == entry_bar:
                    entry_price = opens[i]
                valid_entry = np.isfinite(entry_price) and entry_price > 0
                gain = direction * (closes[i] / entry_price - 1.0) if valid_entry else np.nan
                if not valid_entry or gain >= params.profit_target or i - entry_bar + 1 >= params.max_hold_bars:
                    direction = 0
            elif np.isfinite(closes[i]) and closes[i] > 0:
                direction = 1 if longs[i] else (-1 if shorts[i] else 0)
                if direction:
                    # Record the fill only when its next-bar open becomes observable.
                    entry_bar = i + 1
                    entry_price = np.nan
            signals[i] = direction
        df = pd.DataFrame({"signal": signals, "size": 0.95}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check():
    index = pd.date_range("2020-01-01", periods=5)
    data = pd.DataFrame({"open": [100.0] * 5, "close": [100.0, 103.0, 100.0, 100.0, 100.0]}, index=index)
    ind = pd.DataFrame({"long_entry": [True, False, False, False, False], "short_entry": False}, index=index)
    run = GeneratedStrategy.generate_signals
    assert run(data, ind, None, PressureParams(profit_target=0.02)).data["signal"].tolist() == [0, 1, 0, 0, 0]
    assert run(data, ind, None, PressureParams(profit_target=1.0, max_hold_bars=2)).data["signal"].tolist() == [0, 1, 1, 0, 0]
    ind["short_entry"], ind["long_entry"] = ind["long_entry"].copy(), False
    data["close"] = [100.0, 97.0, 100.0, 100.0, 100.0]
    assert run(data, ind, None, PressureParams(profit_target=0.02)).data["signal"].tolist() == [0, -1, 0, 0, 0]


if __name__ == "__main__":
    _self_check()
