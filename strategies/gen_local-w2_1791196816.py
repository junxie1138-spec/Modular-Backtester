from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class StrategyParams:
    roc_bars: int = 3
    memory_bars: int = 40
    min_correlation: float = 0.20
    forecast_threshold: float = 0.50
    atr_bars: int = 14
    trail_atr: float = 2.5
    hold_bars: int = 8


class GeneratedStrategy(BaseStrategy[StrategyParams]):
    strategy_id = "gen_local-w2_1791196816"

    @classmethod
    def params_type(cls):
        return StrategyParams

    def warmup_bars(self, params: StrategyParams) -> int:
        return max(200, params.roc_bars + params.memory_bars + 2, params.atr_bars + 1)

    def indicators(self, data: pd.DataFrame, params: StrategyParams) -> pd.DataFrame:
        if min(params.roc_bars, params.atr_bars, params.hold_bars) < 1 or params.memory_bars < 3:
            raise ValueError("Lookbacks and holding period must be positive; memory_bars must be at least 3.")
        if not (0 <= params.min_correlation <= 1) or params.forecast_threshold <= 0 or params.trail_atr <= 0:
            raise ValueError("Invalid correlation, forecast threshold, or ATR multiplier.")
        out = pd.DataFrame(index=data.index)
        close = data["close"]
        out["ma200"] = close.rolling(200, min_periods=200).mean()
        out["roc"] = close.pct_change(params.roc_bars, fill_method=None)
        out["acceleration"] = out["roc"].diff()
        out["lagged_acceleration"] = out["acceleration"].shift(1)
        window = params.memory_bars
        out["correlation"] = out["acceleration"].rolling(window, min_periods=window).corr(out["lagged_acceleration"]).clip(-1, 1)
        out["acceleration_rms"] = out["acceleration"].pow(2).rolling(window, min_periods=window).mean().pow(0.5)
        out["forecast"] = (out["correlation"] * out["acceleration"] / out["acceleration_rms"].replace(0, np.nan)).replace([np.inf, -np.inf], np.nan)
        out["previous_close"] = close.shift(1)
        out["true_range"] = pd.concat([
            data["high"] - data["low"],
            (data["high"] - out["previous_close"]).abs(),
            (data["low"] - out["previous_close"]).abs(),
        ], axis=1).max(axis=1)
        out["atr"] = out["true_range"].rolling(params.atr_bars, min_periods=params.atr_bars).mean()
        return out

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: StrategyParams) -> SignalFrame:
        close = data["close"].to_numpy(dtype=float)
        opening = data["open"].to_numpy(dtype=float)
        atr = indicators["atr"].to_numpy(dtype=float)
        ma = indicators["ma200"].to_numpy(dtype=float)
        forecast = indicators["forecast"].to_numpy(dtype=float)
        correlation = indicators["correlation"].to_numpy(dtype=float)
        valid = np.isfinite(close) & np.isfinite(opening) & np.isfinite(atr) & (atr > 0) & np.isfinite(ma) & np.isfinite(forecast) & np.isfinite(correlation)
        candidate = np.zeros(len(data), dtype=int)
        reliable = valid & (np.abs(correlation) >= params.min_correlation)
        candidate[reliable & (close > ma) & (forecast >= params.forecast_threshold)] = 1
        candidate[reliable & (close < ma) & (forecast <= -params.forecast_threshold)] = -1
        signal = np.zeros(len(data), dtype=int)
        position = 0
        age = 0
        pending_entry = False
        high_water = low_water = 0.0
        high_stop = -np.inf
        low_stop = np.inf
        for i in range(len(data)):
            if not valid[i]:
                position = 0
                pending_entry = False
            elif position:
                if pending_entry:
                    high_water = low_water = opening[i]
                    high_stop = -np.inf
                    low_stop = np.inf
                    age = 0
                    pending_entry = False
                age += 1
                high_water = max(high_water, close[i])
                low_water = min(low_water, close[i])
                high_stop = max(high_stop, high_water - params.trail_atr * atr[i])
                low_stop = min(low_stop, low_water + params.trail_atr * atr[i])
                # Highest-close exit applies to both sides; shorts also carry an adverse-price trail.
                stopped = close[i] <= high_stop or (position == -1 and close[i] >= low_stop)
                regime_failed = (position == 1 and close[i] <= ma[i]) or (position == -1 and close[i] >= ma[i])
                if stopped or regime_failed or age >= params.hold_bars:
                    position = 0
            elif candidate[i]:
                position = int(candidate[i])
                pending_entry = True
            signal[i] = position
        df = pd.DataFrame({"signal": signal, "size": 0.95}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check():
    strategy = GeneratedStrategy()
    index = pd.date_range("2020-01-01", periods=5)
    data = pd.DataFrame({"open": 100.0, "high": 111.0, "low": 99.0, "close": [100.0, 104.0, 110.0, 105.0, 105.0], "volume": 1000.0}, index=index)
    features = pd.DataFrame({"ma200": 90.0, "atr": [2.0, 2.0, 2.0, 10.0, 10.0], "forecast": [1.0, 0.0, 0.0, 0.0, 0.0], "correlation": 0.5}, index=index)
    params = StrategyParams(trail_atr=2.0)
    result = strategy.generate_signals(data, features, None, params).data
    assert result["signal"].tolist() == [0, 1, 1, 1, 0]
    assert (result["size"] > 0).all()
    params.hold_bars = 2
    assert strategy.generate_signals(data, features, None, params).data["signal"].tolist() == [0, 1, 1, 0, 0]
    data["close"] = [100.0, 101.0, 105.0, 105.0, 105.0]
    features["ma200"] = 200.0
    features["atr"] = 2.0
    features["forecast"] = [-1.0, 0.0, 0.0, 0.0, 0.0]
    params.hold_bars = 8
    assert strategy.generate_signals(data, features, None, params).data["signal"].tolist() == [0, -1, -1, 0, 0]
    features["forecast"] = np.nan
    assert not strategy.generate_signals(data, features, None, params).data["signal"].any()


if __name__ == "__main__":
    _self_check()
