from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class RangeWaveParams:
    correlation_window: int = 40
    pressure_window: int = 8
    atr_window: int = 14
    forecast_threshold: float = 0.15
    pressure_threshold: float = 0.15
    trail_atr: float = 3.0
    holding_bars: int = 18

    def __post_init__(self):
        if self.correlation_window < 3 or min(self.pressure_window, self.atr_window, self.holding_bars) < 1:
            raise ValueError("Invalid lookback or holding period")
        values = (self.forecast_threshold, self.pressure_threshold, self.trail_atr)
        if not all(np.isfinite(v) for v in values):
            raise ValueError("Thresholds must be finite")
        if self.forecast_threshold < 0 or not 0 <= self.pressure_threshold < 1 or self.trail_atr <= 0:
            raise ValueError("Invalid threshold")


class GeneratedStrategy(BaseStrategy[RangeWaveParams]):
    strategy_id = "gen_local-w2_1791266989"

    @classmethod
    def params_type(cls):
        return RangeWaveParams

    def warmup_bars(self, params: RangeWaveParams) -> int:
        return int(max(params.correlation_window + 2, params.atr_window + 1, params.pressure_window))

    def indicators(self, data: pd.DataFrame, params: RangeWaveParams) -> pd.DataFrame:
        out = pd.DataFrame(index=data.index)
        close = data["close"].where(data["close"] > 0)
        out["relative_range"] = (data["high"] - data["low"]).clip(lower=0) / close
        out["range_change"] = out["relative_range"].diff()
        out["lag_change"] = out["range_change"].shift(1)
        window = params.correlation_window
        out["range_autocorrelation"] = out["range_change"].rolling(window).corr(out["lag_change"]).clip(-1, 1)
        out["change_std"] = out["range_change"].rolling(window).std()
        out["expansion_forecast"] = out["range_autocorrelation"] * out["range_change"] / out["change_std"].replace(0, np.nan)
        out["true_range"] = pd.concat([
            data["high"] - data["low"],
            (data["high"] - close.shift(1)).abs(),
            (data["low"] - close.shift(1)).abs(),
        ], axis=1).max(axis=1)
        out["atr"] = out["true_range"].rolling(params.atr_window).mean()
        volume = data["volume"].clip(lower=0)
        out["signed_volume"] = np.sign(data["close"] - data["open"]) * volume
        out["volume_sum"] = volume.rolling(params.pressure_window).sum()
        out["signed_volume_sum"] = out["signed_volume"].rolling(params.pressure_window).sum()
        out["volume_pressure"] = out["signed_volume_sum"] / out["volume_sum"].replace(0, np.nan)
        return out.replace([np.inf, -np.inf], np.nan)

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame,
                         ctx: StrategyContext, params: RangeWaveParams) -> SignalFrame:
        close = data["close"].to_numpy(dtype=float)
        atr = indicators["atr"].to_numpy(dtype=float)
        forecast = indicators["expansion_forecast"].to_numpy(dtype=float)
        pressure = indicators["volume_pressure"].to_numpy(dtype=float)
        targets = np.zeros(len(data), dtype=int)
        position, age = 0, 0
        extreme = stop = np.nan
        for i in range(len(data)):
            if not np.isfinite(close[i]) or close[i] <= 0 or not np.isfinite(atr[i]) or atr[i] <= 0:
                position, age = 0, 0
                extreme = stop = np.nan
                continue
            if position:
                # First active bar follows the entry decision: its close starts the watermark.
                age += 1
                if position == 1:
                    extreme = close[i] if age == 1 else max(extreme, close[i])
                    candidate = extreme - params.trail_atr * atr[i]
                    stop = candidate if age == 1 else max(stop, candidate)
                    stopped = close[i] <= stop
                else:
                    # Mirror the high-water trail for shorts using the lowest in-trade close.
                    extreme = close[i] if age == 1 else min(extreme, close[i])
                    candidate = extreme + params.trail_atr * atr[i]
                    stop = candidate if age == 1 else min(stop, candidate)
                    stopped = close[i] >= stop
                if stopped or age >= params.holding_bars:
                    position, age = 0, 0
                    extreme = stop = np.nan
            elif np.isfinite(forecast[i]) and np.isfinite(pressure[i]):
                # AND: predicted range expansion and directional volume must both qualify.
                if forecast[i] > params.forecast_threshold:
                    if pressure[i] > params.pressure_threshold:
                        position = 1
                    elif pressure[i] < -params.pressure_threshold:
                        position = -1
            targets[i] = position
        df = pd.DataFrame({"signal": targets, "size": 0.95}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check():
    strategy = GeneratedStrategy()
    params = RangeWaveParams(trail_atr=2.0)
    index = pd.date_range("2020-01-01", periods=8)
    data = pd.DataFrame({"close": [100, 110, 112, 107, 106, 105, 108, 108]}, index=index)
    derived = pd.DataFrame({
        "atr": [1, 1, 3, 3, 1, 1, 1, 1],
        "expansion_forecast": [1.0] * 8,
        "volume_pressure": [1, 1, 1, 1, -1, -1, -1, 0],
    }, index=index)
    signals = strategy.generate_signals(data, derived, None, params).data
    assert signals["signal"].tolist() == [0, 1, 1, 1, 0, -1, -1, 0]
    derived["expansion_forecast"] = 0.0
    assert not strategy.generate_signals(data, derived, None, params).data["signal"].any()
    derived["expansion_forecast"] = np.nan
    assert not strategy.generate_signals(data, derived, None, params).data["signal"].any()


if __name__ == "__main__":
    _self_check()
