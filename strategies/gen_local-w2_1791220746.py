from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class CompressionParams:
    fast_window: int = 5
    slow_window: int = 20
    roc_period: int = 3
    compression_limit: float = 0.60
    acceleration_floor: float = 0.15
    profit_target: float = 0.012
    max_hold: int = 4
    min_size: float = 0.25
    max_size: float = 0.95


class GeneratedStrategy(BaseStrategy[CompressionParams]):
    strategy_id = "gen_local-w2_1791220746"

    @classmethod
    def params_type(cls):
        return CompressionParams

    def warmup_bars(self, params: CompressionParams) -> int:
        return int(max(params.slow_window + 1, params.fast_window + 1, params.roc_period + 2))

    def indicators(self, data: pd.DataFrame, params: CompressionParams) -> pd.DataFrame:
        if not (1 <= params.fast_window < params.slow_window and params.roc_period >= 1):
            raise ValueError("Require 1 <= fast_window < slow_window and roc_period >= 1")
        if not (0 < params.compression_limit < 1 and params.acceleration_floor > 0):
            raise ValueError("Invalid compression or acceleration threshold")
        if not (params.profit_target > 0 and params.max_hold >= 1 and 0 < params.min_size <= params.max_size <= 1):
            raise ValueError("Invalid exit or sizing parameters")
        out = pd.DataFrame(index=data.index)
        close = data["close"]
        out["box_high"] = data["high"].rolling(params.fast_window).max().shift(1)
        out["box_low"] = data["low"].rolling(params.fast_window).min().shift(1)
        out["slow_high"] = data["high"].rolling(params.slow_window).max().shift(1)
        out["slow_low"] = data["low"].rolling(params.slow_window).min().shift(1)
        out["box_width"] = out["box_high"] - out["box_low"]
        out["slow_width"] = out["slow_high"] - out["slow_low"]
        out["compression"] = out["box_width"] / out["slow_width"].replace(0, np.nan)
        out["range_scale"] = ((data["high"] - data["low"]) / close.replace(0, np.nan)).rolling(params.slow_window).mean().shift(1)
        out["roc"] = close.pct_change(params.roc_period, fill_method=None)
        out["previous_roc"] = out["roc"].shift(1)
        out["acceleration"] = out["roc"].diff() / out["range_scale"].replace(0, np.nan)
        out["box_location"] = (close - out["box_low"]) / out["box_width"].replace(0, np.nan)
        out["spring_depth"] = (1 - out["compression"] / params.compression_limit).clip(0, 1)
        out["acceleration_weight"] = out["acceleration"].clip(lower=0) / (1 + out["acceleration"].clip(lower=0))
        out["conviction"] = out["spring_depth"] * out["acceleration_weight"]
        out["entry"] = (
            (out["compression"] < params.compression_limit)
            & (out["previous_roc"] <= 0)
            & (out["roc"] > 0)
            & (out["acceleration"] >= params.acceleration_floor)
            & out["box_location"].between(0.5, 1.0)
        )
        out = out.replace([np.inf, -np.inf], np.nan)
        out["entry"] = out["entry"] & out.drop(columns="entry").notna().all(axis=1)
        return out

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: CompressionParams) -> SignalFrame:
        n = len(data)
        signal = np.zeros(n, dtype=int)
        sizes = np.full(n, params.min_size, dtype=float)
        entries = indicators["entry"].fillna(False).to_numpy(dtype=bool)
        convictions = indicators["conviction"].fillna(0).clip(0, 1).to_numpy(dtype=float)
        opens = data["open"].to_numpy(dtype=float)
        closes = data["close"].to_numpy(dtype=float)
        entry_bar = -1
        entry_price = np.nan
        held_size = params.min_size
        for i in range(n):
            if entry_bar >= 0:
                if i == entry_bar:
                    entry_price = opens[i]
                age = i - entry_bar + 1
                # Targets are observed at the close; exits fill at the next open.
                target_hit = np.isfinite(entry_price) and entry_price > 0 and closes[i] >= entry_price * (1 + params.profit_target)
                if target_hit or age >= params.max_hold:
                    entry_bar = -1
                else:
                    signal[i] = 1
                    sizes[i] = held_size
            elif entries[i]:
                held_size = params.min_size + (params.max_size - params.min_size) * convictions[i]
                signal[i] = 1
                sizes[i] = held_size
                entry_bar = i + 1
        df = pd.DataFrame({"signal": signal, "size": sizes}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = df["size"].shift(1).fillna(params.min_size).astype(float)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check():
    strategy = GeneratedStrategy()
    params = CompressionParams()
    index = pd.date_range("2020-01-01", periods=8)
    data = pd.DataFrame({"open": 100.0, "high": 101.0, "low": 99.0, "close": 100.0, "volume": 1000.0}, index=index)
    indicators = pd.DataFrame({"entry": [True] + [False] * 7, "conviction": 0.5}, index=index)
    result = strategy.generate_signals(data, indicators, None, params).data
    assert result["signal"].tolist() == [0, 1, 1, 1, 1, 0, 0, 0]
    assert np.allclose(result.loc[index[1:5], "size"], 0.60)
    # A gap at entry must set the target basis to the actual entry open.
    data.loc[index[1], ["open", "high", "low", "close"]] = [110.0, 111.0, 109.0, 110.0]
    data.loc[index[2], ["high", "close"]] = [113.0, 112.0]
    result = strategy.generate_signals(data, indicators, None, params).data
    assert result["signal"].tolist() == [0, 1, 1, 0, 0, 0, 0, 0]
    derived = strategy.indicators(data, params)
    assert not derived["entry"].any()
    assert (result["size"] > 0).all()


if __name__ == "__main__":
    _self_check()
