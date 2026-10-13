from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class SlopeInnovationParams:
    slope_window: int = 8
    baseline_window: int = 20
    breakout_window: int = 5
    atr_window: int = 14
    innovation_threshold: float = 0.12
    trailing_atr: float = 1.25
    holding_bars: int = 2


class GeneratedStrategy(BaseStrategy[SlopeInnovationParams]):
    strategy_id = "gen_local-w2_1791284730"

    @classmethod
    def params_type(cls):
        return SlopeInnovationParams

    @staticmethod
    def warmup_bars(params: SlopeInnovationParams) -> int:
        return int(max(200, params.slope_window + params.baseline_window,
                       params.breakout_window + 1, params.atr_window + 1))

    @staticmethod
    def indicators(data: pd.DataFrame, params: SlopeInnovationParams) -> pd.DataFrame:
        if (params.slope_window < 2 or params.baseline_window < 2
                or params.breakout_window < 1 or params.atr_window < 1
                or params.holding_bars not in (1, 2)
                or not np.isfinite(params.innovation_threshold)
                or not np.isfinite(params.trailing_atr)
                or params.innovation_threshold <= 0 or params.trailing_atr <= 0):
            raise ValueError("Invalid strategy parameters")
        close = data["close"].astype(float)
        high = data["high"].astype(float)
        low = data["low"].astype(float)
        out = pd.DataFrame(index=data.index)
        out["previous_close"] = close.shift(1)
        out["true_range"] = pd.concat([
            high - low,
            (high - out["previous_close"]).abs(),
            (low - out["previous_close"]).abs(),
        ], axis=1).max(axis=1, skipna=False)
        out["atr"] = out["true_range"].rolling(params.atr_window).mean()
        out["ma200"] = close.rolling(200).mean()
        out["upper_channel"] = high.rolling(params.breakout_window).max().shift(1)
        out["lower_channel"] = low.rolling(params.breakout_window).min().shift(1)
        n = params.slope_window
        center = (n - 1) / 2.0
        denominator = n * (n * n - 1) / 12.0
        out["slope"] = sum((center - lag) * close.shift(lag)
                           for lag in range(n)) / denominator
        out["latent_slope"] = out["slope"].rolling(params.baseline_window).mean().shift(1)
        out["slope_innovation"] = out["slope"] - out["latent_slope"]
        out["normalized_innovation"] = out["slope_innovation"] / out["atr"].where(out["atr"] > 0)
        required = ["atr", "ma200", "upper_channel", "lower_channel",
                    "slope", "latent_slope", "normalized_innovation"]
        ready = np.isfinite(out[required]).all(axis=1) & np.isfinite(close) & (out["atr"] > 0)
        long_entry = (ready & (close > out["ma200"])
                      & (close > out["upper_channel"]) & (out["slope"] > 0)
                      & (out["normalized_innovation"] > params.innovation_threshold))
        short_entry = (ready & (close < out["ma200"])
                       & (close < out["lower_channel"]) & (out["slope"] < 0)
                       & (out["normalized_innovation"] < -params.innovation_threshold))
        out["entry"] = np.select([long_entry, short_entry], [1, -1], default=0).astype(int)
        return out

    @staticmethod
    def generate_signals(data: pd.DataFrame, indicators: pd.DataFrame,
                         ctx: StrategyContext, params: SlopeInnovationParams) -> SignalFrame:
        close = data["close"].to_numpy(dtype=float)
        atr = indicators["atr"].to_numpy(dtype=float)
        entries = indicators["entry"].fillna(0).to_numpy(dtype=int)
        signals = np.zeros(len(data), dtype=int)
        position = 0
        age = 0
        highest_close = lowest_close = stop = np.nan
        for i in range(len(data)):
            price, volatility = close[i], atr[i]
            if not np.isfinite(price) or not np.isfinite(volatility) or volatility <= 0:
                position = 0
                age = 0
                continue
            if position:
                age += 1
                highest_close = max(highest_close, price)
                lowest_close = min(lowest_close, price)
                if position == 1:
                    # Preserve the ratchet even when ATR expands.
                    stop = max(stop, highest_close - params.trailing_atr * volatility)
                    stopped = price <= stop
                else:
                    # Symmetric trailing stop for short positions.
                    stop = min(stop, lowest_close + params.trailing_atr * volatility)
                    stopped = price >= stop
                if stopped or age >= params.holding_bars:
                    position = 0
                    age = 0
                    continue
            elif entries[i]:
                position = int(entries[i])
                age = 0
                highest_close = lowest_close = price
                stop = price - position * params.trailing_atr * volatility
            signals[i] = position
        df = pd.DataFrame({"signal": signals, "size": 1.0}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


if __name__ == "__main__":
    params = SlopeInnovationParams(trailing_atr=1.0)
    index = pd.date_range("2020-01-01", periods=7)
    indicators = pd.DataFrame({"atr": 1.0, "entry": [1, 0, 0, -1, 0, 0, 0]}, index=index)
    for prices, expected in [
        ([100, 100, 100, 100, 100, 100, 100], [0, 1, 1, 0, -1, -1, 0]),
        ([100, 98, 100, 100, 102, 100, 100], [0, 1, 0, 0, -1, 0, 0]),
    ]:
        data = pd.DataFrame({"close": prices}, index=index)
        result = GeneratedStrategy.generate_signals(data, indicators, None, params)
        assert result.data["signal"].tolist() == expected
        assert (result.data["size"] > 0).all()
