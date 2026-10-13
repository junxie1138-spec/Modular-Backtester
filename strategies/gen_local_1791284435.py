from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class WaveletParams:
    energy_window: int = 20
    correlation_window: int = 40
    percentile_window: int = 126
    entry_percentile: float = 0.80
    atr_window: int = 14
    trail_atr: float = 2.5
    hold_bars: int = 10

    def __post_init__(self):
        if min(self.energy_window, self.correlation_window, self.percentile_window, self.atr_window) < 2:
            raise ValueError("Windows must be at least two.")
        if not 0.0 < self.entry_percentile < 1.0 or self.trail_atr <= 0 or self.hold_bars < 1:
            raise ValueError("Invalid threshold, trail, or holding period.")


class GeneratedStrategy(BaseStrategy[WaveletParams]):
    strategy_id = "gen_local_1791284435"

    @classmethod
    def params_type(cls):
        return WaveletParams

    def warmup_bars(self, params: WaveletParams) -> int:
        return int(max(8 + params.energy_window, params.correlation_window + 5, 21, params.atr_window + 1) + params.percentile_window + 1)

    def indicators(self, data: pd.DataFrame, params: WaveletParams) -> pd.DataFrame:
        out = pd.DataFrame(index=data.index)
        close = data["close"].astype(float)
        out["return"] = np.log(close.where(close > 0)).diff()
        out["volatility"] = out["return"].rolling(20).std()
        previous = close.shift(1)
        out["true_range"] = pd.concat([
            data["high"] - data["low"],
            (data["high"] - previous).abs(),
            (data["low"] - previous).abs(),
        ], axis=1).max(axis=1)
        out["atr"] = out["true_range"].rolling(params.atr_window).mean()
        for scale in (2, 4, 8):
            half = scale // 2
            out[f"recent_{scale}"] = out["return"].rolling(half).mean()
            out[f"older_{scale}"] = out[f"recent_{scale}"].shift(half)
            out[f"haar_{scale}"] = np.sqrt(half / 2.0) * (out[f"recent_{scale}"] - out[f"older_{scale}"])
            out[f"energy_{scale}"] = out[f"haar_{scale}"].pow(2).rolling(params.energy_window).mean()
            out[f"lag_{scale}"] = out["return"].shift(half)
            out[f"rho_{scale}"] = out["return"].rolling(params.correlation_window).corr(out[f"lag_{scale}"]).clip(-1.0, 1.0)
            out[f"predictor_{scale}"] = out["return"].shift(half - 1)
        out["total_energy"] = out["energy_2"] + out["energy_4"] + out["energy_8"]
        denominator = out["total_energy"].where(out["total_energy"] > 0)
        out["forecast"] = 0.0
        for scale in (2, 4, 8):
            out[f"weight_{scale}"] = out[f"energy_{scale}"] / denominator
            out["forecast"] += out[f"weight_{scale}"] * out[f"rho_{scale}"] * out[f"predictor_{scale}"]
        out["strength"] = out["forecast"].abs() / out["volatility"].where(out["volatility"] > 0)
        out["entry_threshold"] = out["strength"].shift(1).rolling(params.percentile_window).quantile(params.entry_percentile)
        return out.replace([np.inf, -np.inf], np.nan)

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: WaveletParams) -> SignalFrame:
        df = pd.DataFrame({"signal": 0, "size": 0.95}, index=data.index)
        close = data["close"].to_numpy(dtype=float)
        atr = indicators["atr"].to_numpy(dtype=float)
        forecast = indicators["forecast"].to_numpy(dtype=float)
        strength = indicators["strength"].to_numpy(dtype=float)
        threshold = indicators["entry_threshold"].to_numpy(dtype=float)
        signals = np.zeros(len(data), dtype=int)
        position, age = 0, 0
        extreme, stop = np.nan, np.nan
        for i in range(len(data)):
            if position:
                age += 1
                if not np.isfinite(close[i]) or not np.isfinite(atr[i]) or atr[i] <= 0:
                    position = 0
                else:
                    if position == 1:
                        extreme = close[i] if np.isnan(extreme) else max(extreme, close[i])
                        candidate = extreme - params.trail_atr * atr[i]
                        stop = candidate if np.isnan(stop) else max(stop, candidate)
                        stopped = close[i] <= stop
                    else:
                        extreme = close[i] if np.isnan(extreme) else min(extreme, close[i])
                        candidate = extreme + params.trail_atr * atr[i]
                        stop = candidate if np.isnan(stop) else min(stop, candidate)
                        stopped = close[i] >= stop
                    if stopped or age >= params.hold_bars:
                        position = 0
                # An exit remains flat for this decision bar.
            elif (
                np.isfinite(close[i]) and np.isfinite(atr[i]) and atr[i] > 0
                and np.isfinite(forecast[i]) and np.isfinite(strength[i])
                and np.isfinite(threshold[i]) and strength[i] > threshold[i]
                and forecast[i] != 0
            ):
                position = 1 if forecast[i] > 0 else -1
                age = 0
                extreme, stop = np.nan, np.nan
            signals[i] = position
        df["signal"] = signals
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check():
    strategy = GeneratedStrategy()
    params = WaveletParams()
    index = pd.date_range("2020-01-01", periods=6)
    for direction in (1, -1):
        close = np.array([100, 110, 108, 104, 100, 100], dtype=float)
        if direction == -1:
            close = 200 - close
        data = pd.DataFrame({"open": close, "high": close + 1, "low": close - 1, "close": close, "volume": 1000.0}, index=index)
        ind = pd.DataFrame({"atr": 2.0, "forecast": [direction, 0, 0, 0, 0, 0], "strength": [2, 0, 0, 0, 0, 0], "entry_threshold": 1.0}, index=index)
        result = strategy.generate_signals(data, ind, None, params).data
        assert result["signal"].tolist() == [0, direction, direction, direction, 0, 0]
        assert (result["size"] > 0).all()
        ind["atr"] = 100.0
        short_params = WaveletParams(hold_bars=2)
        result = strategy.generate_signals(data, ind, None, short_params).data
        assert result["signal"].tolist() == [0, direction, direction, 0, 0, 0]


if __name__ == "__main__":
    _self_check()
