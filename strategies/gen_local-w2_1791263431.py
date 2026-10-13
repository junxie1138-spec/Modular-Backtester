from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class GeneratedParams:
    excursion_bars: int = 6
    atr_bars: int = 20
    fast_bars: int = 2
    min_depth_atr: float = 1.5
    phase_gap: float = 0.15
    profit_target: float = 0.012
    max_hold: int = 4

    def __post_init__(self):
        if not (self.excursion_bars >= 2 and self.atr_bars >= 2 and 1 <= self.fast_bars <= self.excursion_bars):
            raise ValueError("Invalid indicator windows")
        if not (self.min_depth_atr > 0 and 0 < self.phase_gap < 1 and 0 < self.profit_target < 1 and 3 <= self.max_hold <= 5):
            raise ValueError("Invalid entry or exit parameters")


class GeneratedStrategy(BaseStrategy[GeneratedParams]):
    strategy_id = "gen_local-w2_1791263431"

    @classmethod
    def params_type(cls):
        return GeneratedParams

    def warmup_bars(self, params: GeneratedParams) -> int:
        return int(max(params.atr_bars + params.excursion_bars + 1, params.excursion_bars + 1, params.fast_bars + 1))

    def indicators(self, data: pd.DataFrame, params: GeneratedParams) -> pd.DataFrame:
        out = pd.DataFrame(index=data.index)
        close = data["close"]
        out["previous_close"] = close.shift(1)
        out["true_range"] = pd.concat([
            data["high"] - data["low"],
            (data["high"] - out["previous_close"]).abs(),
            (data["low"] - out["previous_close"]).abs(),
        ], axis=1).max(axis=1)
        out["baseline_atr"] = out["true_range"].rolling(params.atr_bars, min_periods=params.atr_bars).mean().shift(params.excursion_bars)
        out["fast_atr"] = out["true_range"].rolling(params.fast_bars, min_periods=params.fast_bars).mean()
        out["peak_range"] = out["true_range"].rolling(params.excursion_bars, min_periods=params.excursion_bars).max()
        out["anchor"] = close.shift(params.excursion_bars)
        out["trough"] = close.rolling(params.excursion_bars, min_periods=params.excursion_bars).min()
        out["depth"] = out["anchor"] - out["trough"]
        out["depth_atr"] = out["depth"] / out["baseline_atr"].where(out["baseline_atr"] > 0)
        out["repair"] = (close - out["trough"]) / out["depth"].where(out["depth"] > 0)
        out["excess_range"] = out["peak_range"] - out["baseline_atr"]
        out["release"] = ((out["peak_range"] - out["fast_atr"]) / out["excess_range"].where(out["excess_range"] > 0)).clip(0.0, 1.0)
        out["phase_difference"] = out["release"] - out["repair"]
        out["close_change"] = close.diff()
        out = out.replace([np.inf, -np.inf], np.nan)
        out["eligible"] = (
            (out["depth_atr"] >= params.min_depth_atr)
            & (out["peak_range"] >= 1.4 * out["baseline_atr"])
            & out["repair"].between(0.15, 0.65)
        ).fillna(False)
        out["entry_long"] = (out["eligible"] & (out["phase_difference"] >= params.phase_gap) & (out["close_change"] > 0)).fillna(False)
        out["entry_short"] = (out["eligible"] & (out["phase_difference"] <= -params.phase_gap) & (out["close_change"] < 0)).fillna(False)
        out["strength"] = (out["phase_difference"].abs() / 0.5).clip(0.0, 1.0).fillna(0.0)
        return out

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: GeneratedParams) -> SignalFrame:
        n = len(data)
        signal = np.zeros(n, dtype=int)
        size = np.full(n, 0.25, dtype=float)
        longs = indicators["entry_long"].fillna(False).to_numpy(dtype=bool)
        shorts = indicators["entry_short"].fillna(False).to_numpy(dtype=bool)
        strength = indicators["strength"].fillna(0.0).to_numpy(dtype=float)
        opens = data["open"].to_numpy(dtype=float)
        closes = data["close"].to_numpy(dtype=float)
        direction, fill_bar = 0, -1
        entry_price, held_size = np.nan, 0.25
        for i in range(n):
            if direction:
                if i == fill_bar:
                    entry_price = opens[i]
                valid_entry = np.isfinite(entry_price) and entry_price > 0
                profit = direction * (closes[i] / entry_price - 1.0) if valid_entry and np.isfinite(closes[i]) else -np.inf
                # Exit decisions at this close execute at the following open.
                if not valid_entry or profit >= params.profit_target or i - fill_bar + 1 >= params.max_hold:
                    direction = 0
                    continue
            elif np.isfinite(closes[i]) and closes[i] > 0:
                direction = 1 if longs[i] else (-1 if shorts[i] else 0)
                if direction:
                    fill_bar = i + 1
                    entry_price = np.nan
                    held_size = 0.25 + 0.70 * float(np.clip(strength[i], 0.0, 1.0))
            signal[i] = direction
            if direction:
                size[i] = held_size
        df = pd.DataFrame({"signal": signal, "size": size}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = df["size"].shift(1).fillna(0.25).astype(float)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check():
    strategy = GeneratedStrategy()
    params = GeneratedParams()
    index = pd.date_range("2020-01-01", periods=14)
    data = pd.DataFrame({"open": 100.0, "high": 103.0, "low": 89.0, "close": 100.0, "volume": 1000.0}, index=index)
    data.loc[index[0], "close"] = 90.0
    data.loc[index[2], "close"] = 102.0
    data.loc[index[6], "close"] = 98.0
    ind = pd.DataFrame({"entry_long": False, "entry_short": False, "strength": 0.4}, index=index)
    ind.loc[index[[0, 8]], "entry_long"] = True
    ind.loc[index[4], "entry_short"] = True
    ctx = StrategyContext(symbol="SPY", timeframe="1d", warmup_bars=0)
    result = strategy.generate_signals(data, ind, ctx, params).data
    assert result["signal"].tolist() == [0, 1, 1, 0, 0, -1, -1, 0, 0, 1, 1, 1, 1, 0]
    assert np.allclose(result.loc[result["signal"] != 0, "size"], 0.53)
    assert (result["size"] > 0).all()
    assert strategy.generate_signals(data.iloc[:7], ind.iloc[:7], ctx, params).data.equals(result.iloc[:7])
    prices = 100.0 + np.sin(np.arange(80) / 3.0)
    sample = pd.DataFrame({"open": prices, "high": prices + 1.0, "low": prices - 1.0, "close": prices, "volume": 1000.0}, index=pd.date_range("2021-01-01", periods=80))
    full = strategy.indicators(sample, params)
    assert strategy.indicators(sample.iloc[:50], params).equals(full.iloc[:50])
    assert not full["entry_long"].iloc[:strategy.warmup_bars(params) - 1].any()
    assert not full["entry_short"].iloc[:strategy.warmup_bars(params) - 1].any()


if __name__ == "__main__":
    _self_check()
