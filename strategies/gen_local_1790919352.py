from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class GeneratedParams:
    lookback: int = 32
    stop_atr: float = 2.5


class GeneratedStrategy(BaseStrategy[GeneratedParams]):
    strategy_id = "gen_local_1790919352"

    @classmethod
    def params_type(cls):
        return GeneratedParams

    def warmup_bars(self, params: GeneratedParams) -> int:
        return max(params.lookback, 21, 15)

    def indicators(self, data: pd.DataFrame, params: GeneratedParams) -> pd.DataFrame:
        if params.lookback < 4 or not np.isfinite(params.stop_atr) or params.stop_atr <= 0:
            raise ValueError("lookback must be >= 4 and stop_atr must be positive.")
        out = pd.DataFrame(index=data.index)
        close = data["close"].where(data["close"] > 0)
        out["log_close"] = np.log(close)
        out["log_return"] = out["log_close"].diff()
        out["volatility"] = out["log_return"].rolling(20, min_periods=20).std(ddof=0) * np.sqrt(252.0)
        previous = close.shift(1)
        out["true_range"] = pd.concat(
            [data["high"] - data["low"], (data["high"] - previous).abs(),
             (data["low"] - previous).abs()], axis=1
        ).max(axis=1)
        out["atr"] = out["true_range"].rolling(14, min_periods=14).mean()
        upper = np.zeros(len(data))
        lower = np.zeros(len(data))
        maximum = np.full(len(data), -np.inf)
        minimum = np.full(len(data), np.inf)
        # Endpoint visibility: an edge clears every intervening chord slope.
        for lag in range(1, params.lookback):
            slope = ((out["log_close"].shift(lag) - out["log_close"]) / lag).to_numpy()
            finite = np.isfinite(slope)
            if lag >= 3:
                upper += finite & (slope > maximum)
                lower += finite & (slope < minimum)
            maximum = np.where(finite, np.maximum(maximum, slope), maximum)
            minimum = np.where(finite, np.minimum(minimum, slope), minimum)
        complete = out["log_close"].rolling(params.lookback).count().eq(params.lookback)
        out["upper_degree"] = pd.Series(upper, index=data.index).where(complete)
        out["lower_degree"] = pd.Series(lower, index=data.index).where(complete)
        total = out["upper_degree"] + out["lower_degree"]
        out["topology"] = ((out["upper_degree"] - out["lower_degree"]) / total.where(total > 0)).fillna(0.0)
        out["displacement"] = out["log_close"] - out["log_close"].shift(params.lookback - 1)
        out["ready"] = complete & out["atr"].gt(0) & out["volatility"].gt(0)
        out["candidate"] = np.select(
            [out["ready"] & out["topology"].gt(0.25) & out["displacement"].gt(0),
             out["ready"] & out["topology"].lt(-0.25) & out["displacement"].lt(0)],
            [1, -1], default=0
        ).astype(int)
        out["target_size"] = (0.12 / out["volatility"].where(out["volatility"] > 0)).clip(0.01, 1.0).fillna(1.0)
        return out

    def generate_signals(
        self, data: pd.DataFrame, indicators: pd.DataFrame,
        ctx: StrategyContext, params: GeneratedParams,
    ) -> SignalFrame:
        n = len(data)
        signal = np.zeros(n, dtype=int)
        sizes = np.ones(n, dtype=float)
        close = data["close"].to_numpy(dtype=float)
        opening = data["open"].to_numpy(dtype=float)
        candidate = indicators["candidate"].to_numpy(dtype=int)
        atr = indicators["atr"].to_numpy(dtype=float)
        target_size = indicators["target_size"].to_numpy(dtype=float)
        direction = 0
        entry_price = np.nan
        entry_bar = 0
        stop_distance = 0.0
        position_size = 1.0
        for i in range(n):
            if direction:
                # Resolve the previous close's entry at this bar's observed open.
                if np.isnan(entry_price):
                    entry_price = opening[i]
                    entry_bar = i
                valid = np.isfinite(entry_price) and entry_price > 0 and np.isfinite(close[i])
                stopped = valid and direction * (close[i] - entry_price) < -stop_distance
                if not valid or stopped or i - entry_bar + 1 >= 18:
                    direction = 0
            elif candidate[i] and np.isfinite(atr[i]) and atr[i] > 0:
                direction = int(candidate[i])
                entry_price = np.nan
                stop_distance = params.stop_atr * atr[i]
                position_size = float(target_size[i])
            signal[i] = direction
            sizes[i] = position_size if direction else 1.0
        df = pd.DataFrame({"signal": signal, "size": sizes}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = df["size"].shift(1).fillna(1.0).astype(float)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


if __name__ == "__main__":
    strategy = GeneratedStrategy()
    params = GeneratedParams()
    index = pd.date_range("2020-01-01", periods=40, freq="B")
    data = pd.DataFrame(
        {"open": 100.0, "high": 101.0, "low": 99.0, "close": 100.0, "volume": 1000.0},
        index=index,
    )
    features = pd.DataFrame({"candidate": 0, "atr": 2.0, "target_size": 0.5}, index=index)
    ctx = StrategyContext(symbol="SPY", timeframe="1d", warmup_bars=0)
    for direction in (1, -1):
        features["candidate"] = 0
        features.iloc[0, features.columns.get_loc("candidate")] = direction
        held = strategy.generate_signals(data, features, ctx, params).data
        assert held["signal"].iloc[0] == 0
        assert held["signal"].iloc[1:19].eq(direction).all()
        assert held["signal"].iloc[19:].eq(0).all()
        shocked = data.copy()
        shocked.iloc[1, shocked.columns.get_loc("close")] = 100.0 - direction * 6.0
        stopped = strategy.generate_signals(shocked, features, ctx, params).data
        assert stopped["signal"].iloc[1] == direction
        assert stopped["signal"].iloc[2:].eq(0).all()
    flat = strategy.generate_signals(data, strategy.indicators(data, params), ctx, params).data
    assert flat["signal"].eq(0).all() and flat["size"].gt(0).all()
