from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class FractionalPressureParams:
    fractional_order: float = 0.45
    fractional_window: int = 40
    envelope_window: int = 8
    breakout_threshold: float = 0.5
    hold_bars: int = 8
    min_size: float = 0.25
    max_size: float = 0.95


class GeneratedStrategy(BaseStrategy[FractionalPressureParams]):
    strategy_id = "gen_local_1791280831"

    @classmethod
    def params_type(cls) -> type[FractionalPressureParams]:
        return FractionalPressureParams

    def warmup_bars(self, params: FractionalPressureParams) -> int:
        if not (0.0 < params.fractional_order < 1.0):
            raise ValueError("fractional_order must be between zero and one")
        if params.fractional_window < 2 or params.envelope_window < 2:
            raise ValueError("Lookback windows must be at least two")
        if params.hold_bars < 1 or not np.isfinite(params.breakout_threshold) or params.breakout_threshold <= 0.0:
            raise ValueError("Holding period and breakout threshold must be positive")
        if not (0.0 < params.min_size <= params.max_size <= 1.0):
            raise ValueError("Sizes must satisfy 0 < min_size <= max_size <= 1")
        return int(params.fractional_window + 2 * params.envelope_window + 1)

    def indicators(self, data: pd.DataFrame, params: FractionalPressureParams) -> pd.DataFrame:
        self.warmup_bars(params)
        close = data["close"].astype(float).replace([np.inf, -np.inf], np.nan)
        close = close.where(close > 0.0)
        weights = np.ones(params.fractional_window, dtype=float)
        for k in range(1, params.fractional_window):
            weights[k] = -weights[k - 1] * (params.fractional_order - k + 1) / k
        fractional_close = close.copy()
        for k in range(1, params.fractional_window):
            fractional_close = fractional_close + weights[k] * close.shift(k)

        window = params.envelope_window
        floor = fractional_close.rolling(window, min_periods=window).min().shift(1)
        ceiling = fractional_close.rolling(window, min_periods=window).max().shift(1)
        old_floor = floor.shift(window)
        old_ceiling = ceiling.shift(window)
        innovation = fractional_close.diff()
        noise = innovation.rolling(window, min_periods=window).std(ddof=0).shift(1).replace(0.0, np.nan)
        strength = ((fractional_close - ceiling) / noise).replace([np.inf, -np.inf], np.nan)
        eligible = (
            ((floor - old_floor) > 0.25 * noise)
            & ((ceiling - old_ceiling) <= 0.25 * noise)
            & (strength >= params.breakout_threshold)
        ).fillna(False)
        return pd.DataFrame({
            "fractional_close": fractional_close,
            "innovation": innovation,
            "floor": floor,
            "ceiling": ceiling,
            "old_floor": old_floor,
            "old_ceiling": old_ceiling,
            "noise": noise,
            "strength": strength,
            "eligible": eligible,
        }, index=data.index).replace([np.inf, -np.inf], np.nan).fillna(0.0)

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: FractionalPressureParams,
    ) -> SignalFrame:
        self.warmup_bars(params)
        eligible = indicators["eligible"].fillna(False).to_numpy(dtype=bool)
        strength = indicators["strength"].replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy(dtype=float)
        signals = np.zeros(len(data), dtype=int)
        sizes = np.full(len(data), params.min_size, dtype=float)
        exit_bar = -1
        entry_size = params.min_size
        for i in range(len(data)):
            # An exit decision always produces a flat bar before another entry.
            if i == exit_bar:
                continue
            if i > exit_bar:
                if not eligible[i]:
                    continue
                exit_bar = i + params.hold_bars
                entry_size = float(params.min_size + (params.max_size - params.min_size) * np.clip(strength[i] / 3.0, 0.0, 1.0))
            signals[i] = 1
            sizes[i] = entry_size
        df = pd.DataFrame({"signal": signals, "size": sizes}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = df["size"].shift(1).fillna(params.min_size).astype(float)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check() -> None:
    strategy = GeneratedStrategy()
    params = FractionalPressureParams(hold_bars=3)
    index = pd.date_range("2020-01-01", periods=12, freq="B")
    data = pd.DataFrame({"open": 100.0, "high": 101.0, "low": 99.0, "close": 100.0, "volume": 1000.0}, index=index)
    indicators = pd.DataFrame({"eligible": True, "strength": np.linspace(0.5, 3.0, len(data))}, index=index)
    result = strategy.generate_signals(data, indicators, None, params).data
    assert result["signal"].tolist() == [0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1]
    assert result["size"].iloc[1:4].nunique() == 1
    assert result["size"].iloc[5] > result["size"].iloc[1]
    assert result["size"].between(params.min_size, params.max_size).all()
    indicators["eligible"] = False
    assert strategy.generate_signals(data, indicators, None, params).data["signal"].eq(0).all()
    assert not strategy.indicators(data, params)["eligible"].astype(bool).any()


if __name__ == "__main__":
    _self_check()
