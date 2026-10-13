from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class StrategyParams:
    fractional_order: float = 0.45
    memory: int = 40
    horizon: int = 18
    baseline: int = 63
    compression_limit: float = 0.9
    fractional_threshold: float = 0.2
    pressure_threshold: float = 0.12
    size: float = 1.0


class GeneratedStrategy(BaseStrategy[StrategyParams]):
    strategy_id = "gen_local_1791243996"

    @classmethod
    def params_type(cls):
        return StrategyParams

    def warmup_bars(self, params: StrategyParams) -> int:
        return int(max(params.memory + max(params.horizon, params.baseline) - 1,
                       max(params.horizon, params.baseline) + 1))

    def indicators(self, data: pd.DataFrame, params: StrategyParams) -> pd.DataFrame:
        if not (params.memory >= 2 and params.horizon >= 2
                and params.baseline >= params.horizon
                and 0 < params.fractional_order < 1):
            raise ValueError("Invalid fractional order or lookbacks")
        controls = [params.compression_limit, params.fractional_threshold,
                    params.pressure_threshold, params.size]
        if not np.isfinite(controls).all() or min(controls) <= 0:
            raise ValueError("Thresholds and size must be finite and positive")
        out = pd.DataFrame(index=data.index)
        close = data["close"].astype(float)
        high = data["high"].astype(float)
        low = data["low"].astype(float)
        fractional = close.copy()
        weight = 1.0
        # Fixed finite fractional filter; all terms use present or past closes.
        for lag in range(1, params.memory):
            weight *= (lag - 1 - params.fractional_order) / lag
            fractional = fractional + weight * close.shift(lag)
        out["fractional_close"] = fractional
        out["fractional_fast"] = fractional.rolling(params.horizon).mean()
        out["fractional_slow"] = fractional.rolling(params.baseline).mean()
        out["fractional_std"] = fractional.rolling(params.baseline).std(ddof=0)
        out["fractional_direction"] = (
            (out["fractional_fast"] - out["fractional_slow"])
            / out["fractional_std"].replace(0.0, np.nan)
        )
        out["bar_range"] = (high - low).clip(lower=0.0)
        out["true_range"] = pd.concat(
            [out["bar_range"], (high - close.shift(1)).abs(),
             (low - close.shift(1)).abs()], axis=1
        ).max(axis=1)
        out["range_fast"] = out["true_range"].rolling(params.horizon).mean()
        out["range_slow"] = out["true_range"].rolling(params.baseline).mean()
        out["compression"] = (
            out["range_fast"] / out["range_slow"].replace(0.0, np.nan)
        )
        overlap_high = pd.concat([high, high.shift(1)], axis=1).min(axis=1, skipna=False)
        overlap_low = pd.concat([low, low.shift(1)], axis=1).max(axis=1, skipna=False)
        out["overlap"] = (overlap_high - overlap_low).clip(lower=0.0)
        out["overlap_weight"] = (
            out["overlap"] / out["bar_range"].replace(0.0, np.nan)
        ).clip(0.0, 1.0)
        out["closing_location"] = (
            (2.0 * close - high - low) / out["bar_range"].replace(0.0, np.nan)
        ).clip(-1.0, 1.0)
        out["weighted_pressure"] = out["overlap_weight"] * out["closing_location"]
        out["pressure_sum"] = out["weighted_pressure"].rolling(params.horizon).sum()
        out["overlap_sum"] = out["overlap_weight"].rolling(params.horizon).sum()
        out["pressure"] = out["pressure_sum"] / out["overlap_sum"].replace(0.0, np.nan)
        return out.replace([np.inf, -np.inf], np.nan)

    def generate_signals(
        self, data: pd.DataFrame, indicators: pd.DataFrame,
        ctx: StrategyContext, params: StrategyParams,
    ) -> SignalFrame:
        compressed = indicators["compression"].le(params.compression_limit)
        long_entry = (
            compressed
            & indicators["fractional_direction"].gt(params.fractional_threshold)
            & indicators["pressure"].gt(params.pressure_threshold)
        )
        short_entry = (
            compressed
            & indicators["fractional_direction"].lt(-params.fractional_threshold)
            & indicators["pressure"].lt(-params.pressure_threshold)
        )
        requests = np.select([long_entry, short_entry], [1, -1], default=0)
        signals = np.zeros(len(data), dtype=int)
        position = 0
        for i, request in enumerate(requests):
            if position == 0:
                position = int(request)
            elif request == -position:
                position = 0
            signals[i] = position
        df = pd.DataFrame(index=data.index)
        df["signal"] = signals
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = float(params.size)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


if __name__ == "__main__":
    strategy = GeneratedStrategy()
    params = StrategyParams()
    index = pd.date_range("2020-01-01", periods=7)
    data = pd.DataFrame(index=index)
    indicators = pd.DataFrame({
        "compression": [0.8] * 7,
        "fractional_direction": [1.0, 1.0, np.nan, -1.0, -1.0, 0.0, 1.0],
        "pressure": [0.5, 0.5, np.nan, -0.5, -0.5, 0.0, 0.5],
    }, index=index)
    ctx = StrategyContext(symbol="SPY", timeframe="1d",
                          warmup_bars=strategy.warmup_bars(params))
    result = strategy.generate_signals(data, indicators, ctx, params).data
    assert result["signal"].tolist() == [0, 1, 1, 1, 0, -1, -1]
    assert result["size"].gt(0).all()
