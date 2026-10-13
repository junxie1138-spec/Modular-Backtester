from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class Params:
    fd_order: float = 0.45
    fd_window: int = 48
    correlation_window: int = 30
    rebound_threshold: float = 0.15
    pressure_window: int = 3
    pressure_threshold: float = 0.10
    atr_window: int = 14
    trail_atr: float = 2.4
    max_hold: int = 8


class GeneratedStrategy(BaseStrategy[Params]):
    strategy_id = "gen_local-w2_1791228886"

    @classmethod
    def params_type(cls):
        return Params

    def warmup_bars(self, params: Params) -> int:
        return int(max(params.fd_window + params.correlation_window,
                       params.pressure_window, params.atr_window + 1))

    def indicators(self, data: pd.DataFrame, params: Params) -> pd.DataFrame:
        if not (0.0 < params.fd_order < 1.0
                and params.fd_window >= 2 and params.correlation_window >= 3
                and params.pressure_window >= 1 and params.atr_window >= 1
                and 5 <= params.max_hold <= 10 and params.trail_atr > 0
                and params.rebound_threshold > 0
                and 0.0 <= params.pressure_threshold < 1.0):
            raise ValueError("Invalid strategy parameters")
        out = pd.DataFrame(index=data.index)
        close = data["close"].astype(float).replace([np.inf, -np.inf], np.nan)
        # ponytail: finite fractional memory; increase fd_window for longer memory.
        fd = close.copy()
        weight = 1.0
        for lag in range(1, params.fd_window):
            weight *= (lag - 1 - params.fd_order) / lag
            fd = fd + weight * close.shift(lag)
        out["fractional_close"] = fd
        out["fractional_lag"] = fd.shift(1)
        window = params.correlation_window
        out["fractional_mean"] = fd.rolling(window).mean()
        out["fractional_std"] = fd.rolling(window).std(ddof=0)
        out["autocorrelation"] = fd.rolling(window).corr(out["fractional_lag"]).clip(-1.0, 1.0)
        out["fractional_z"] = (fd - out["fractional_mean"]) / out["fractional_std"].replace(0.0, np.nan)
        out["rebound_score"] = out["autocorrelation"] * out["fractional_z"]
        candle_range = (data["high"] - data["low"]).replace(0.0, np.nan)
        out["close_location"] = ((2.0 * close - data["high"] - data["low"]) / candle_range).clip(-1.0, 1.0)
        volume = data["volume"].astype(float).where(data["volume"] >= 0)
        out["signed_volume"] = out["close_location"] * volume
        out["volume_sum"] = volume.rolling(params.pressure_window).sum()
        out["pressure"] = out["signed_volume"].rolling(params.pressure_window).sum() / out["volume_sum"].replace(0.0, np.nan)
        out["previous_close"] = close.shift(1)
        out["true_range"] = pd.concat([
            data["high"] - data["low"],
            (data["high"] - out["previous_close"]).abs(),
            (data["low"] - out["previous_close"]).abs(),
        ], axis=1).max(axis=1, skipna=False)
        out["atr"] = out["true_range"].rolling(params.atr_window).mean()
        return out.replace([np.inf, -np.inf], np.nan)

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame,
                         ctx: StrategyContext, params: Params) -> SignalFrame:
        eligible = ((indicators["autocorrelation"] < -0.10)
                    & (indicators["rebound_score"] > params.rebound_threshold)
                    & (indicators["pressure"] > params.pressure_threshold)).fillna(False).to_numpy()
        close = data["close"].to_numpy(dtype=float)
        atr = indicators["atr"].to_numpy(dtype=float)
        signals = np.zeros(len(data), dtype=int)
        active = False
        age = 0
        high_water = -np.inf
        stop = -np.inf
        for i in range(len(data)):
            if not (np.isfinite(close[i]) and np.isfinite(atr[i]) and atr[i] > 0):
                active = False
                age = 0
                high_water = stop = -np.inf
                continue
            if active:
                # Entry fills today when yesterday's target first became one.
                age += 1
                high_water = max(high_water, close[i])
                stop = max(stop, high_water - params.trail_atr * atr[i])
                if close[i] <= stop or age >= params.max_hold:
                    active = False
                    age = 0
                    high_water = stop = -np.inf
                    continue
            elif i >= self.warmup_bars(params) - 1 and eligible[i]:
                active = True
                age = 0
                high_water = stop = -np.inf
            signals[i] = int(active)
        df = pd.DataFrame({"signal": signals, "size": 0.95}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check() -> None:
    strategy = GeneratedStrategy()
    params = Params(fd_window=2, correlation_window=3, atr_window=1, pressure_window=1)
    index = pd.date_range("2020-01-01", periods=10)
    close = np.array([100., 100., 100., 100., 100., 105., 104., 102., 101., 100.])
    data = pd.DataFrame({"open": close, "high": close + 1, "low": close - 1,
                         "close": close, "volume": 1000.}, index=index)
    indicators = pd.DataFrame({"autocorrelation": -0.5, "rebound_score": 1.0,
                               "pressure": 1.0, "atr": [1., 1., 1., 1., 1., 1., 4., 4., 1., 1.]},
                              index=index)
    indicators.loc[index[5]:, "pressure"] = 0.0
    ctx = StrategyContext(symbol="SPY", timeframe="1d", warmup_bars=5)
    result = strategy.generate_signals(data, indicators, ctx, params).data
    assert result["signal"].tolist() == [0, 0, 0, 0, 0, 1, 1, 1, 0, 0]
    assert (result["size"] > 0).all()
    assert strategy.indicators(data.iloc[:8], params).equals(
        strategy.indicators(data, params).iloc[:8])


if __name__ == "__main__":
    _self_check()
