from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class StrategyParams:
    slope_window: int = 12
    innovation_window: int = 3
    compression_window: int = 8
    compression_limit: float = 0.38
    innovation_threshold: float = 0.10
    atr_window: int = 14
    trailing_atr: float = 2.8
    max_holding_bars: int = 18


class GeneratedStrategy(BaseStrategy[StrategyParams]):
    strategy_id = "gen_local_1790983680"

    @classmethod
    def params_type(cls) -> type[StrategyParams]:
        return StrategyParams

    @staticmethod
    def warmup_bars(params: StrategyParams) -> int:
        return int(max(
            max(2, params.slope_window) + max(1, params.innovation_window),
            max(1, params.compression_window) + 1,
            max(1, params.atr_window) + 1,
        ))

    @staticmethod
    def indicators(data: pd.DataFrame, params: StrategyParams) -> pd.DataFrame:
        out = pd.DataFrame(index=data.index)
        close = data["close"].astype(float)
        previous = close.shift(1)
        true_range = pd.concat([
            data["high"] - data["low"],
            (data["high"] - previous).abs(),
            (data["low"] - previous).abs(),
        ], axis=1).max(axis=1)
        out["true_range"] = true_range
        out["atr"] = true_range.rolling(max(1, params.atr_window)).mean()

        length = max(2, params.slope_window)
        coordinates = np.arange(length, dtype=float)
        weights = coordinates - coordinates.mean()
        values = close.to_numpy(dtype=float)
        slope_values = np.full(len(data), np.nan)
        if len(data) >= length:
            slope_values[length - 1:] = np.convolve(
                values, weights[::-1], mode="valid"
            ) / np.square(weights).sum()
        out["slope"] = slope_values
        out["trend_mean"] = close.rolling(length).mean()
        out["fitted_endpoint"] = out["trend_mean"] + out["slope"] * (length - 1) / 2.0
        out["forecast"] = (out["fitted_endpoint"] + out["slope"]).shift(1)
        out["innovation"] = (close - out["forecast"]) / out["atr"].replace(0.0, np.nan)
        out["persistent_innovation"] = out["innovation"].rolling(
            max(1, params.innovation_window)
        ).mean()

        compression_length = max(1, params.compression_window)
        out["envelope_width"] = (
            data["high"].rolling(compression_length).max()
            - data["low"].rolling(compression_length).min()
        )
        out["traveled_range"] = true_range.rolling(compression_length).sum()
        out["compression_ratio"] = (
            out["envelope_width"] / out["traveled_range"].replace(0.0, np.nan)
        )
        out["trend_agrees"] = (
            (out["slope"] > 0.0)
            & (out["persistent_innovation"] > params.innovation_threshold)
        ).fillna(False)
        out["compression_agrees"] = (
            out["compression_ratio"] <= params.compression_limit
        ).fillna(False)
        return out.replace([np.inf, -np.inf], np.nan)

    @staticmethod
    def generate_signals(
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: StrategyParams,
    ) -> SignalFrame:
        df = pd.DataFrame(index=data.index)
        entries = (
            indicators["trend_agrees"].fillna(False).astype(bool)
            & indicators["compression_agrees"].fillna(False).astype(bool)
            & indicators["atr"].gt(0.0)
        ).to_numpy(dtype=bool)
        closes = data["close"].to_numpy(dtype=float)
        atrs = indicators["atr"].to_numpy(dtype=float)
        signals = np.zeros(len(data), dtype=int)
        active = False
        high_water = np.nan
        stop = np.nan
        age = 0
        multiplier = max(0.01, float(params.trailing_atr))
        holding_limit = max(1, int(params.max_holding_bars))

        for i in range(len(data)):
            if active:
                age += 1
                if np.isfinite(closes[i]):
                    # The first active bar is the fill bar; exclude the decision bar.
                    high_water = closes[i] if np.isnan(high_water) else max(high_water, closes[i])
                    if np.isfinite(atrs[i]) and atrs[i] > 0.0:
                        candidate = high_water - multiplier * atrs[i]
                        stop = candidate if np.isnan(stop) else max(stop, candidate)
                    if np.isfinite(stop) and closes[i] <= stop:
                        active = False
                if age >= holding_limit:
                    active = False
            elif entries[i] and np.isfinite(closes[i]) and np.isfinite(atrs[i]):
                active = True
                high_water = np.nan
                stop = np.nan
                age = 0
            signals[i] = int(active)

        df["signal"] = signals
        df["size"] = 0.95
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


if __name__ == "__main__":
    index = pd.date_range("2020-01-01", periods=6)
    data = pd.DataFrame({
        "open": [100., 103., 106., 101., 100., 100.],
        "high": [101., 104., 107., 102., 101., 101.],
        "low": [99., 102., 105., 100., 99., 99.],
        "close": [100., 103., 106., 101., 100., 100.],
        "volume": [1000.] * 6,
    }, index=index)
    derived = pd.DataFrame({
        "trend_agrees": [True, False, False, False, False, False],
        "compression_agrees": [True] * 6,
        "atr": [1.] * 6,
    }, index=index)
    result = GeneratedStrategy.generate_signals(data, derived, None, StrategyParams())
    assert result.data["signal"].tolist() == [0, 1, 1, 1, 0, 0]
    assert result.data["size"].gt(0.0).all()
    assert GeneratedStrategy.indicators(data.iloc[:0], StrategyParams()).index.equals(data.iloc[:0].index)
