from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class DrawdownCapacityParams:
    peak_window: int = 20
    atr_window: int = 14
    depth_atr: float = 2.0
    stop_atr: float = 1.25


class GeneratedStrategy(BaseStrategy[DrawdownCapacityParams]):
    strategy_id = "gen_local_1790938355"

    @classmethod
    def params_type(cls):
        return DrawdownCapacityParams

    def warmup_bars(self, params):
        return max(200, params.peak_window + 2, params.atr_window + 2)

    def indicators(self, data, params):
        if params.peak_window < 2 or params.atr_window < 1:
            raise ValueError("Lookback windows must be positive; peak_window must be at least two.")
        if not np.isfinite(params.depth_atr) or params.depth_atr <= 0:
            raise ValueError("depth_atr must be positive and finite.")
        if not np.isfinite(params.stop_atr) or params.stop_atr <= 0:
            raise ValueError("stop_atr must be positive and finite.")
        out = pd.DataFrame(index=data.index)
        close = data["close"]
        previous_close = close.shift(1)
        out["ma200"] = close.rolling(200, min_periods=200).mean()
        out["peak"] = close.shift(1).rolling(params.peak_window).max()
        out["true_range"] = pd.concat(
            [data["high"] - data["low"],
             (data["high"] - previous_close).abs(),
             (data["low"] - previous_close).abs()], axis=1
        ).max(axis=1, skipna=False)
        out["atr"] = out["true_range"].rolling(params.atr_window).mean()
        # Lagged volatility keeps today's move from diluting its own depth.
        out["depth"] = (out["peak"] - close).clip(lower=0) / out["atr"].shift(1).replace(0, np.nan)
        out["previous_depth"] = out["depth"].shift(1)
        out["stable_peak"] = out["peak"].eq(out["peak"].shift(1))
        return out

    def generate_signals(self, data, indicators, ctx: StrategyContext, params):
        close = data["close"]
        depth = indicators["depth"]
        previous = indicators["previous_depth"]
        valid = (
            np.isfinite(indicators[["ma200", "atr", "depth", "previous_depth"]]).all(axis=1)
            & np.isfinite(close) & (close > 0) & (indicators["atr"] > 0)
            & indicators["stable_peak"]
        )
        long_entry = valid & (close > indicators["ma200"]) & (previous >= params.depth_atr) & (depth < params.depth_atr)
        short_entry = valid & (close < indicators["ma200"]) & (previous <= params.depth_atr) & (depth > params.depth_atr)
        entries = np.select([long_entry, short_entry], [1, -1], default=0)
        prices = close.to_numpy(dtype=float)
        opens = data["open"].to_numpy(dtype=float)
        atr = indicators["atr"].to_numpy(dtype=float)
        raw = np.zeros(len(data), dtype=int)
        side = 0
        decision_bar = -1
        stop_price = np.nan
        distance = 0.0
        for i in range(len(data)):
            if side:
                # The decision was made yesterday; observe its fill open only now.
                if i == decision_bar + 1:
                    if not np.isfinite(opens[i]) or opens[i] <= 0:
                        side = 0
                        continue
                    stop_price = opens[i] - side * distance
                stopped = not np.isfinite(prices[i]) or side * (prices[i] - stop_price) < 0
                if stopped or i - decision_bar >= 2:
                    side = 0
                raw[i] = side
            elif entries[i]:
                side = int(entries[i])
                decision_bar = i
                distance = params.stop_atr * atr[i]
                raw[i] = side
        df = pd.DataFrame(index=data.index)
        df["signal"] = raw
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = 1.0
        return SignalFrame(data=df, signal_column="signal", size_column="size")


if __name__ == "__main__":
    # Runnable check: shifted entry, actual-open fixed stops, shorts, two-session expiry.
    index = pd.date_range("2020-01-01", periods=8)
    data = pd.DataFrame({
        "open": [100., 105., 101., 100., 100., 101., 100., 100.],
        "high": [106.] * 8, "low": [95.] * 8,
        "close": [100., 103., 100., 100., 100., 100., 100., 100.],
        "volume": [1000.] * 8
    }, index=index)
    ind = pd.DataFrame({
        "ma200": [90., 90., 90., 110., 110., 90., 90., 90.],
        "atr": [1.] * 8,
        "depth": [1., 1., 1., 3., 3., 1., 1., 1.],
        "previous_depth": [3., 1., 1., 1., 3., 3., 1., 1.],
        "stable_peak": [True] * 8
    }, index=index)
    strategy = GeneratedStrategy()
    result = strategy.generate_signals(data, ind, None, DrawdownCapacityParams()).data
    assert result["signal"].tolist() == [0, 1, 0, 0, -1, -1, 0, 0]
    ind.loc[index[5], "depth"] = 3.
    ind.loc[index[5], "previous_depth"] = 3.
    data.loc[index[4], "close"] = 102.
    result = strategy.generate_signals(data, ind, None, DrawdownCapacityParams()).data
    assert result["signal"].tolist() == [0, 1, 0, 0, -1, 0, 0, 0]
