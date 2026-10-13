from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class ShockwaveParams:
    fast_window: int = 3
    slow_window: int = 20
    atr_window: int = 14
    expansion: float = 1.25
    min_shock: float = 0.5
    stop_atr: float = 1.2
    hold_bars: int = 2


class GeneratedStrategy(BaseStrategy[ShockwaveParams]):
    strategy_id = "gen_local_1791284002"

    @classmethod
    def params_type(cls):
        return ShockwaveParams

    def warmup_bars(self, params: ShockwaveParams) -> int:
        return int(max(params.fast_window + 2, params.slow_window + 2, params.atr_window + 1))

    def indicators(self, data: pd.DataFrame, params: ShockwaveParams) -> pd.DataFrame:
        if not (2 <= params.fast_window < params.slow_window and params.atr_window >= 1):
            raise ValueError("Require 2 <= fast_window < slow_window and atr_window >= 1.")
        if not (params.expansion > 1 and params.min_shock > 0 and params.stop_atr > 0 and params.hold_bars in (1, 2)):
            raise ValueError("Require expansion > 1, positive thresholds, and hold_bars in (1, 2).")
        out = pd.DataFrame(index=data.index)
        out["return"] = data["close"].pct_change(fill_method=None)
        previous = data["close"].shift(1)
        out["tr"] = pd.concat([
            data["high"] - data["low"],
            (data["high"] - previous).abs(),
            (data["low"] - previous).abs(),
        ], axis=1).max(axis=1)
        out["atr"] = out["tr"].rolling(params.atr_window, min_periods=params.atr_window).mean()
        out["fast_vol"] = out["return"].rolling(params.fast_window, min_periods=params.fast_window).std(ddof=0)
        out["slow_vol"] = out["return"].rolling(params.slow_window, min_periods=params.slow_window).std(ddof=0)
        out["ratio"] = out["fast_vol"] / out["slow_vol"].replace(0.0, np.nan)
        out["front_growth"] = out["ratio"].diff()
        out["shock"] = data["close"].diff() / out["atr"].replace(0.0, np.nan)
        out["regime"] = np.where(out["front_growth"] > 0, 1, -1)
        out["direction"] = np.sign(out["shock"]) * out["regime"]
        out["strength"] = ((out["ratio"] - 1.0) / (params.expansion - 1.0) * out["shock"].abs()).clip(0.0, 1.0)
        out["ready"] = np.isfinite(out[["atr", "ratio", "front_growth", "shock", "strength"]]).all(axis=1) & (out["atr"] > 0)
        out["candidate"] = out["direction"].where(
            out["ready"] & (out["ratio"] >= params.expansion) & (out["shock"].abs() >= params.min_shock),
            0,
        ).fillna(0).astype(int)
        return out

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: ShockwaveParams) -> SignalFrame:
        n = len(data)
        signals = np.zeros(n, dtype=int)
        sizes = np.full(n, 0.2, dtype=float)
        opens = data["open"].to_numpy(dtype=float)
        closes = data["close"].to_numpy(dtype=float)
        atrs = indicators["atr"].to_numpy(dtype=float)
        candidates = indicators["candidate"].fillna(0).to_numpy(dtype=int)
        strengths = indicators["strength"].fillna(0.0).clip(0.0, 1.0).to_numpy(dtype=float)
        position = 0
        entry_bar = -1
        stop = np.nan
        frozen_atr = 0.0
        entry_size = 0.2
        for i in range(n):
            if position:
                age = i - entry_bar
                if age == 1:
                    # The preceding close's decision fills at this bar's open.
                    stop = opens[i] - position * params.stop_atr * frozen_atr
                stopped = position * (closes[i] - stop) < 0
                if stopped or age >= params.hold_bars or not np.isfinite(closes[i]) or not np.isfinite(stop):
                    position = 0
                    continue
                signals[i] = position
                sizes[i] = entry_size
            elif candidates[i] and np.isfinite(closes[i]) and np.isfinite(atrs[i]) and atrs[i] > 0:
                position = int(candidates[i])
                entry_bar = i
                frozen_atr = atrs[i]
                entry_size = 0.2 + 0.75 * strengths[i]
                signals[i] = position
                sizes[i] = entry_size
        df = pd.DataFrame({"signal": signals, "size": sizes}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = df["size"].shift(1).fillna(0.2).astype(float)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


if __name__ == "__main__":
    index = pd.date_range("2020-01-01", periods=7)
    data = pd.DataFrame({"open": 100.0, "high": 104.0, "low": 96.0, "close": 100.0, "volume": 1000.0}, index=index)
    ind = pd.DataFrame({"atr": [1.0, 10.0, 10.0, 1.0, 10.0, 10.0, 10.0], "candidate": [1, 0, 0, -1, 0, 0, 0], "strength": [0.4, 0, 0, 0.8, 0, 0, 0]}, index=index)
    strategy = GeneratedStrategy()
    params = ShockwaveParams()
    ctx = StrategyContext(symbol="SPY", timeframe="1d", warmup_bars=0)
    timed = strategy.generate_signals(data, ind, ctx, params).data
    assert timed["signal"].tolist() == [0, 1, 1, 0, -1, -1, 0]
    assert np.isclose(timed["size"].iloc[1], 0.5) and np.isclose(timed["size"].iloc[4], 0.8)
    data.loc[index[1], "close"] = 98.0
    data.loc[index[4], "close"] = 102.0
    stopped = strategy.generate_signals(data, ind, ctx, params).data
    assert stopped["signal"].tolist() == [0, 1, 0, 0, -1, 0, 0]
    derived = strategy.indicators(data, params)
    assert derived.index.equals(data.index) and derived["candidate"].eq(0).all()
    assert strategy.warmup_bars(params) == 22
