from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class ElasticParams:
    frac_order: float = 0.6
    frac_window: int = 32
    memory_window: int = 40
    reversal_corr: float = 0.1
    forecast_threshold: float = 0.3
    pressure_window: int = 3
    pressure_threshold: float = 0.1
    atr_window: int = 14
    stop_atr: float = 2.0
    hold_bars: int = 8


class GeneratedStrategy(BaseStrategy[ElasticParams]):
    strategy_id = "gen_local-w2_1791281221"

    @classmethod
    def params_type(cls):
        return ElasticParams

    def warmup_bars(self, params):
        return int(max(params.frac_window + params.memory_window,
                       params.atr_window + 1, params.pressure_window))

    def indicators(self, data, params):
        if not (0.0 < params.frac_order < 1.0):
            raise ValueError("frac_order must be between zero and one")
        if min(params.frac_window, params.memory_window,
               params.pressure_window, params.atr_window) < 2:
            raise ValueError("indicator windows must be at least two")
        close = data["close"].astype(float)
        weights = np.ones(params.frac_window, dtype=float)
        for j in range(1, params.frac_window):
            weights[j] = -weights[j - 1] * (params.frac_order - j + 1) / j
        # ponytail: fixed-width fractional filter; increase frac_window for a longer memory.
        fractional = close * weights[0]
        for j in range(1, params.frac_window):
            fractional = fractional + weights[j] * close.shift(j)
        mean = fractional.rolling(params.memory_window).mean()
        std = fractional.rolling(params.memory_window).std(ddof=0).replace(0.0, np.nan)
        z = (fractional - mean) / std
        rho = fractional.rolling(params.memory_window).corr(fractional.shift(1))
        forecast = rho * z
        daily_range = (data["high"] - data["low"]).replace(0.0, np.nan)
        signed_volume = ((close - data["open"]) / daily_range).clip(-1.0, 1.0) * data["volume"]
        volume_sum = data["volume"].rolling(params.pressure_window).sum().replace(0.0, np.nan)
        pressure = signed_volume.rolling(params.pressure_window).sum() / volume_sum
        previous_close = close.shift(1)
        true_range = pd.concat([
            data["high"] - data["low"],
            (data["high"] - previous_close).abs(),
            (data["low"] - previous_close).abs(),
        ], axis=1).max(axis=1)
        atr = true_range.rolling(params.atr_window).mean()
        return pd.DataFrame({
            "fractional_close": fractional,
            "fractional_mean": mean,
            "fractional_std": std,
            "fractional_z": z,
            "autocorrelation": rho,
            "forecast": forecast,
            "daily_range": daily_range,
            "signed_volume": signed_volume,
            "volume_sum": volume_sum,
            "pressure": pressure,
            "previous_close": previous_close,
            "true_range": true_range,
            "atr": atr,
        }, index=data.index).replace([np.inf, -np.inf], np.nan)

    def generate_signals(self, data, indicators, ctx, params):
        if params.stop_atr <= 0 or params.hold_bars < 1:
            raise ValueError("stop_atr and hold_bars must be positive")
        if min(params.reversal_corr, params.forecast_threshold,
               params.pressure_threshold) < 0:
            raise ValueError("entry thresholds must be nonnegative")
        rho = indicators["autocorrelation"]
        forecast = indicators["forecast"]
        pressure = indicators["pressure"]
        reverting = rho < -params.reversal_corr
        long_entry = reverting & (forecast > params.forecast_threshold) & (pressure > params.pressure_threshold)
        short_entry = reverting & (forecast < -params.forecast_threshold) & (pressure < -params.pressure_threshold)
        desired = np.select([long_entry, short_entry], [1, -1], default=0)
        valid = np.isfinite(indicators[["autocorrelation", "forecast", "pressure", "atr"]]).all(axis=1)
        valid &= np.isfinite(data[["open", "close"]]).all(axis=1)
        valid &= indicators["atr"] > 0.0
        closes = data["close"].to_numpy(dtype=float)
        opens = data["open"].to_numpy(dtype=float)
        atrs = indicators["atr"].to_numpy(dtype=float)
        ready = valid.to_numpy(dtype=bool)
        signals = np.zeros(len(data), dtype=int)
        side = 0
        pending = False
        stop = np.nan
        entry_bar = -1
        frozen_atr = np.nan
        for i in range(len(data)):
            if not ready[i]:
                side = 0
                pending = False
                continue
            if pending:
                # The preceding close requested entry; this bar's open is now known.
                entry_bar = i
                stop = opens[i] - side * params.stop_atr * frozen_atr
                pending = False
            if side:
                stopped = (side == 1 and closes[i] < stop) or (side == -1 and closes[i] > stop)
                expired = i - entry_bar + 1 >= params.hold_bars
                if stopped or expired:
                    side = 0
                    signals[i] = 0
                    continue
            if side == 0 and desired[i] != 0:
                side = int(desired[i])
                frozen_atr = atrs[i]
                pending = True
            signals[i] = side
        df = pd.DataFrame({"signal": signals, "size": 0.95}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


if __name__ == "__main__":
    index = pd.date_range("2020-01-01", periods=9)
    data = pd.DataFrame({
        "open": 100.0, "high": 104.0, "low": 96.0,
        "close": [100.0, 100.0, 97.0, 100.0, 100.0, 103.0, 100.0, 100.0, 100.0],
        "volume": 1000.0,
    }, index=index)
    indicators = pd.DataFrame({
        "autocorrelation": -0.5,
        "forecast": [1.0, 1.0, 0.0, -1.0, -1.0, 0.0, 0.0, 0.0, 0.0],
        "pressure": [0.5, 0.5, 0.0, -0.5, -0.5, 0.0, 0.0, 0.0, 0.0],
        "atr": 1.0,
    }, index=index)
    result = GeneratedStrategy().generate_signals(data, indicators, None, ElasticParams())
    assert result.data["signal"].tolist() == [0, 1, 1, 0, -1, -1, 0, 0, 0]
    indicators["forecast"] = 1.0
    indicators["pressure"] = 0.5
    data["close"] = 100.0
    result = GeneratedStrategy().generate_signals(data, indicators, None, ElasticParams(hold_bars=3))
    assert result.data["signal"].tolist() == [0, 1, 1, 1, 0, 1, 1, 1, 0]
