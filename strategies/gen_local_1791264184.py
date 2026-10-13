from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class QuarterOpeningParams:
    ma_window: int = 40
    atr_window: int = 14
    entry_z: float = 1.0
    stop_atr: float = 2.2
    hold_bars: int = 8
    min_size: float = 0.25
    max_size: float = 0.95


class GeneratedStrategy(BaseStrategy[QuarterOpeningParams]):
    strategy_id = "gen_local_1791264184"

    @classmethod
    def params_type(cls) -> type[QuarterOpeningParams]:
        return QuarterOpeningParams

    def warmup_bars(self, params: QuarterOpeningParams) -> int:
        if params.ma_window < 2 or params.atr_window < 1:
            raise ValueError("Invalid indicator windows")
        if params.entry_z <= 0 or params.stop_atr <= 0 or params.hold_bars < 1:
            raise ValueError("Thresholds and holding period must be positive")
        if not 0 < params.min_size <= params.max_size <= 1:
            raise ValueError("Sizes must satisfy 0 < min_size <= max_size <= 1")
        return int(max(params.ma_window, params.atr_window + 1))

    def indicators(self, data: pd.DataFrame, params: QuarterOpeningParams) -> pd.DataFrame:
        self.warmup_bars(params)
        close = data["close"].astype(float)
        average = close.rolling(params.ma_window, min_periods=params.ma_window).mean()
        dispersion = close.rolling(params.ma_window, min_periods=params.ma_window).std(ddof=0)
        previous_close = close.shift(1)
        ranges = pd.concat([
            data["high"] - data["low"],
            (data["high"] - previous_close).abs(),
            (data["low"] - previous_close).abs(),
        ], axis=1)
        true_range = ranges.max(axis=1).where(previous_close.notna())
        atr = true_range.rolling(params.atr_window, min_periods=params.atr_window).mean()
        quarter_key = pd.Series(data.index.year * 4 + (data.index.month - 1) // 3, index=data.index)
        quarter_session = quarter_key.groupby(quarter_key, sort=False).cumcount()
        calendar_gate = pd.Series(
            ((data.index.month - 1) % 3 == 0) & (data.index.day <= 7),
            index=data.index,
        ) & quarter_session.lt(3)
        return pd.DataFrame({
            "average": average,
            "dispersion": dispersion,
            "z": ((close - average) / dispersion.where(dispersion > 0)).replace([np.inf, -np.inf], np.nan),
            "true_range": true_range,
            "atr": atr,
            "quarter_session": quarter_session,
            "calendar_gate": calendar_gate,
        }, index=data.index)

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: QuarterOpeningParams) -> SignalFrame:
        self.warmup_bars(params)
        count = len(data)
        signals = np.zeros(count, dtype=int)
        sizes = np.full(count, params.min_size, dtype=float)
        close = data["close"].to_numpy(dtype=float)
        opening = data["open"].to_numpy(dtype=float)
        z = indicators["z"].to_numpy(dtype=float)
        atr = indicators["atr"].to_numpy(dtype=float)
        gate = indicators["calendar_gate"].fillna(False).to_numpy(dtype=bool)
        side = 0
        entry = np.nan
        distance = 0.0
        age = 0
        exposure = params.min_size
        for i in range(count):
            if side:
                # Anchor at the execution bar's open, after the prior close's decision.
                if np.isnan(entry):
                    entry = opening[i]
                    if not np.isfinite(entry) or entry <= 0:
                        side = 0
                        entry = np.nan
                if side:
                    age += 1
                    stopped = np.isfinite(close[i]) and side * (close[i] - entry) < -distance
                    if stopped or age >= params.hold_bars:
                        side = 0
                        entry = np.nan
            elif gate[i] and np.isfinite(z[i]) and np.isfinite(atr[i]) and atr[i] > 0 and np.isfinite(close[i]) and close[i] > 0:
                if abs(z[i]) >= params.entry_z:
                    side = -1 if z[i] > 0 else 1
                    entry = np.nan
                    age = 0
                    # Freeze decision-bar ATR; the stop never trails.
                    distance = params.stop_atr * atr[i]
                    strength = np.clip((abs(z[i]) - params.entry_z) / (2.0 * params.entry_z), 0.0, 1.0)
                    exposure = float(params.min_size + (params.max_size - params.min_size) * strength)
            signals[i] = side
            sizes[i] = exposure if side else params.min_size
        df = pd.DataFrame({"signal": signals, "size": sizes}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = df["size"].shift(1).fillna(params.min_size).astype(float)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


if __name__ == "__main__":
    strategy = GeneratedStrategy()
    dates = pd.bdate_range("2025-03-31", periods=6)
    for direction, prices, holding in [
        (1, [100, 110, 100, 95, 100, 100], 8),
        (-1, [100, 90, 100, 105, 100, 100], 8),
        (1, [100, 100, 100, 100, 100, 100], 2),
    ]:
        data = pd.DataFrame({"open": 100.0, "high": 112.0, "low": 88.0, "close": prices, "volume": 1000.0}, index=dates)
        derived = pd.DataFrame({"z": [0, -2 * direction, 0, 0, 0, 0], "atr": 2.0, "calendar_gate": [False, True, False, False, False, False]}, index=dates)
        result = strategy.generate_signals(data, derived, None, QuarterOpeningParams(hold_bars=holding)).data
        assert result["signal"].tolist() == [0, 0, direction, direction, 0, 0]
        assert np.allclose(result["size"].iloc[2:4], 0.6)
