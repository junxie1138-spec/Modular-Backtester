from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class WaveParams:
    frac_order: float = 0.45
    frac_window: int = 32
    z_window: int = 40
    entry_z: float = 1.5
    atr_window: int = 14
    stop_atr: float = 2.0
    hold_days: int = 4


class GeneratedStrategy(BaseStrategy[WaveParams]):
    strategy_id = "gen_local-w2_1790941052"

    @classmethod
    def params_type(cls):
        return WaveParams

    def warmup_bars(self, params: WaveParams) -> int:
        return int(max(params.frac_window + params.z_window + 2, params.atr_window + 1))

    def indicators(self, data: pd.DataFrame, params: WaveParams) -> pd.DataFrame:
        if not (0.0 < params.frac_order < 1.0 and params.frac_window >= 2
                and params.z_window >= 3 and params.atr_window >= 2
                and params.entry_z > 0.5 and params.stop_atr > 0.0
                and 3 <= params.hold_days <= 5):
            raise ValueError("Invalid fractional-wave parameters")
        close = data["close"].astype(float)
        # Fixed-width fractional difference: w[k] = w[k-1] * (k-1-d) / k.
        fractional = close.copy()
        weight = 1.0
        for lag in range(1, params.frac_window):
            weight *= (lag - 1.0 - params.frac_order) / lag
            fractional = fractional + weight * close.shift(lag)
        mean = fractional.rolling(params.z_window).mean()
        std = fractional.rolling(params.z_window).std(ddof=0).replace(0.0, np.nan)
        previous = close.shift(1)
        true_range = pd.concat([
            data["high"] - data["low"],
            (data["high"] - previous).abs(),
            (data["low"] - previous).abs(),
        ], axis=1).max(axis=1)
        result = pd.DataFrame({
            "fractional_close": fractional,
            "fractional_change": fractional.diff(),
            "fractional_mean": mean,
            "fractional_std": std,
            "true_range": true_range,
            "z": (fractional - mean) / std,
            "atr": true_range.rolling(params.atr_window).mean(),
        }, index=data.index)
        return result.replace([np.inf, -np.inf], np.nan)

    def generate_signals(
        self, data: pd.DataFrame, indicators: pd.DataFrame,
        ctx: StrategyContext, params: WaveParams,
    ) -> SignalFrame:
        z = indicators["z"].to_numpy(dtype=float)
        change = indicators["fractional_change"].to_numpy(dtype=float)
        atr = indicators["atr"].to_numpy(dtype=float)
        close = data["close"].to_numpy(dtype=float)
        opens = data["open"].to_numpy(dtype=float)
        signals = np.zeros(len(data), dtype=int)
        sizes = np.ones(len(data), dtype=float)
        side, entry_bar = 0, -1
        stop, distance, exposure = np.nan, np.nan, 1.0
        for i in range(len(data)):
            if side:
                # Anchor once to the next session's observed open; never trail.
                if i == entry_bar:
                    stop = opens[i] - side * distance
                stopped = np.isfinite(close[i]) and side * (close[i] - stop) < 0.0
                expired = i - entry_bar + 1 >= params.hold_days
                if stopped or expired or not np.isfinite(stop):
                    side = 0
            elif i >= self.warmup_bars(params) - 1:
                if (np.isfinite(z[i - 1]) and np.isfinite(z[i])
                        and np.isfinite(change[i]) and np.isfinite(atr[i])
                        and atr[i] > 0.0 and np.isfinite(close[i])):
                    if z[i - 1] <= -params.entry_z and z[i - 1] < z[i] < -0.5 and change[i] > 0.0:
                        side = 1
                    elif z[i - 1] >= params.entry_z and 0.5 < z[i] < z[i - 1] and change[i] < 0.0:
                        side = -1
                    if side:
                        entry_bar = i + 1
                        distance = params.stop_atr * atr[i]
                        exposure = float(np.clip(abs(z[i - 1]) / (2.0 * params.entry_z), 0.25, 1.0))
            signals[i] = side
            sizes[i] = exposure if side else 1.0
        df = pd.DataFrame({"signal": signals, "size": sizes}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = df["size"].shift(1).fillna(1.0).astype(float)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


if __name__ == "__main__":
    # Runnable check: entry-open anchoring, symmetric fixed stops, expiry, sizing.
    params = WaveParams(frac_window=2, z_window=3, atr_window=2)
    n = 20
    index = pd.date_range("2020-01-01", periods=n)
    data = pd.DataFrame({
        "open": 100.0, "high": 111.0, "low": 95.0,
        "close": 100.0, "volume": 1000.0,
    }, index=index)
    ind = pd.DataFrame({"z": 0.0, "fractional_change": 0.0, "atr": 1.0}, index=index)
    ind.iloc[6:8, 0] = [-2.0, -1.5]
    ind.iloc[7, 1] = 1.0
    data.iloc[8, data.columns.get_loc("open")] = 110.0
    data.iloc[8:10, data.columns.get_loc("close")] = [109.0, 107.0]
    ind.iloc[11:13, 0] = [2.0, 1.5]
    ind.iloc[12, 1] = -1.0
    data.iloc[14, data.columns.get_loc("close")] = 103.0
    strategy = GeneratedStrategy()
    ctx = StrategyContext("SPY", "1d", strategy.warmup_bars(params))
    output = strategy.generate_signals(data, ind, ctx, params).data
    assert output["signal"].tolist() == [0] * 8 + [1, 1, 0, 0, 0, -1, -1] + [0] * 5
    assert np.isclose(output["size"].iloc[8], 2.0 / 3.0)
    data["open"] = data["close"] = 100.0
    ind.iloc[11:13, 0] = 0.0
    output = strategy.generate_signals(data, ind, ctx, params).data
    assert output["signal"].tolist() == [0] * 8 + [1] * 4 + [0] * 8
