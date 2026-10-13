from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class Params:
    fractional_order: float = 0.4
    fractional_window: int = 32
    normalization_window: int = 20
    queue_window: int = 4
    capacity: float = 2.5
    atr_window: int = 14
    breakeven_pct: float = 0.012
    trail_atr: float = 2.0
    holding_bars: int = 4


class GeneratedStrategy(BaseStrategy[Params]):
    strategy_id = "gen_local_1791232167"

    @classmethod
    def params_type(cls) -> type[Params]:
        return Params

    def warmup_bars(self, params: Params) -> int:
        return int(max(200, params.fractional_window + params.normalization_window + params.queue_window, params.atr_window + 1))

    def indicators(self, data: pd.DataFrame, params: Params) -> pd.DataFrame:
        if not (0.0 < params.fractional_order < 1.0):
            raise ValueError("fractional_order must be between zero and one")
        if min(params.fractional_window, params.normalization_window, params.queue_window, params.atr_window) < 2:
            raise ValueError("lookbacks must be at least two")
        if not (0.0 < params.capacity < params.queue_window):
            raise ValueError("capacity must be positive and below queue_window")
        if params.trail_atr <= 0.0 or params.breakeven_pct <= 0.0 or not 3 <= params.holding_bars <= 5:
            raise ValueError("invalid exit parameters")
        close = data["close"].astype(float)
        out = pd.DataFrame(index=data.index)
        weights = np.ones(params.fractional_window, dtype=float)
        for lag in range(1, params.fractional_window):
            weights[lag] = -weights[lag - 1] * (params.fractional_order - lag + 1) / lag
        fractional = close * weights[0]
        for lag in range(1, params.fractional_window):
            fractional = fractional + weights[lag] * close.shift(lag)
        out["fractional_close"] = fractional
        prior = fractional.shift(1)
        out["fractional_mean"] = prior.rolling(params.normalization_window).mean()
        out["fractional_std"] = prior.rolling(params.normalization_window).std(ddof=0)
        out["fractional_z"] = (fractional - out["fractional_mean"]) / out["fractional_std"].replace(0.0, np.nan)
        out["pressure"] = (out["fractional_z"] - 0.5).clip(lower=0.0, upper=1.0)
        out["backlog"] = out["pressure"].rolling(params.queue_window).sum()
        out["ma200"] = close.rolling(200).mean()
        previous_close = close.shift(1)
        out["true_range"] = pd.concat([
            data["high"] - data["low"],
            (data["high"] - previous_close).abs(),
            (data["low"] - previous_close).abs(),
        ], axis=1).max(axis=1)
        out["atr"] = out["true_range"].rolling(params.atr_window).mean()
        out["entry"] = (
            (out["backlog"] >= params.capacity)
            & (out["backlog"].shift(1) < params.capacity)
            & (fractional > fractional.shift(1))
            & (close > out["ma200"])
            & (out["atr"] > 0.0)
        ).fillna(False)
        return out

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: Params) -> SignalFrame:
        df = pd.DataFrame({"signal": np.zeros(len(data), dtype=int), "size": 0.95}, index=data.index)
        entry_flags = indicators["entry"].fillna(False).to_numpy(dtype=bool)
        atrs = indicators["atr"].to_numpy(dtype=float)
        prices = data[["open", "high", "low", "close"]].to_numpy(dtype=float)
        pending = False
        active = False
        armed = False
        entry = stop = peak = 0.0
        entry_bar = 0
        for i, (opening, high, low, close) in enumerate(prices):
            if pending:
                pending = False
                if np.isfinite(opening) and opening > 0.0 and i > 0 and np.isfinite(atrs[i - 1]) and atrs[i - 1] > 0.0:
                    active = True
                    armed = False
                    entry = peak = opening
                    stop = entry - params.trail_atr * atrs[i - 1]
                    entry_bar = i
            if active:
                if not np.isfinite([opening, high, low, close, atrs[i]]).all() or atrs[i] <= 0.0:
                    active = False
                    continue
                # Stops are close-time exit decisions filled at the next open.
                # Test the previous stop before incorporating today's high.
                exit_now = low <= stop
                if not exit_now:
                    peak = max(peak, high)
                    armed = armed or high >= entry * (1.0 + params.breakeven_pct)
                    if armed:
                        stop = max(stop, entry, peak - params.trail_atr * atrs[i])
                    exit_now = close <= stop
                if exit_now or i - entry_bar + 1 >= params.holding_bars:
                    active = False
                    continue
                df.iat[i, 0] = 1
            elif entry_flags[i] and np.isfinite(atrs[i]) and atrs[i] > 0.0:
                pending = True
                df.iat[i, 0] = 1
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


if __name__ == "__main__":
    index = pd.date_range("2020-01-01", periods=7, freq="D")
    sample = pd.DataFrame({
        "open": [100, 100, 101, 103, 102, 102, 102],
        "high": [101, 102, 104, 103, 103, 103, 103],
        "low": [99, 99, 100.5, 101, 101, 101, 101],
        "close": [100, 101, 103, 102, 102, 102, 102],
        "volume": [1000] * 7,
    }, index=index)
    derived = pd.DataFrame({"entry": [True] + [False] * 6, "atr": [1.0] * 7}, index=index)
    result = GeneratedStrategy().generate_signals(sample, derived, None, Params())
    assert result.data["signal"].tolist() == [0, 1, 1, 1, 0, 0, 0]
    assert (result.data["size"] > 0.0).all()
