from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class AnalogParams:
    ma_window: int = 20
    shape_bars: int = 5
    history_bars: int = 252
    horizon_bars: int = 8
    atr_window: int = 14
    bandwidth: float = 0.6
    entry_z: float = -0.6
    min_forecast: float = 0.004
    atr_mult: float = 2.0


class GeneratedStrategy(BaseStrategy[AnalogParams]):
    strategy_id = "gen_local_1791212815"

    @classmethod
    def params_type(cls) -> type[AnalogParams]:
        return AnalogParams

    def warmup_bars(self, params: AnalogParams) -> int:
        if (params.ma_window < 2 or params.shape_bars < 2
                or params.horizon_bars < 1 or params.atr_window < 1
                or params.history_bars < params.shape_bars + params.horizon_bars
                or not np.isfinite(params.bandwidth) or params.bandwidth <= 0
                or not np.isfinite(params.atr_mult) or params.atr_mult <= 0):
            raise ValueError("Invalid analog lookbacks or stop parameters")
        return int(max(200 + params.history_bars,
                       params.ma_window + params.shape_bars - 1 + params.history_bars,
                       params.horizon_bars + 1 + params.history_bars,
                       params.atr_window + 1))

    def indicators(self, data: pd.DataFrame, params: AnalogParams) -> pd.DataFrame:
        self.warmup_bars(params)
        out = pd.DataFrame(index=data.index)
        close = data["close"]
        out["ma"] = close.rolling(params.ma_window).mean()
        out["std"] = close.rolling(params.ma_window).std(ddof=0)
        out["z"] = ((close - out["ma"]) / out["std"].replace(0.0, np.nan))
        out["regime_ma"] = close.rolling(200).mean()
        previous = close.shift(1)
        out["tr"] = pd.concat(
            [data["high"] - data["low"],
             (data["high"] - previous).abs(),
             (data["low"] - previous).abs()], axis=1).max(axis=1)
        out["atr"] = out["tr"].rolling(params.atr_window).mean()
        out["matured_return"] = close.pct_change(params.horizon_bars, fill_method=None)
        for offset in range(1, params.shape_bars):
            out[f"z_lag_{offset}"] = out["z"].shift(offset)
        shape = out[["z"] + [f"z_lag_{i}" for i in range(1, params.shape_bars)]].to_numpy()
        prices = close.to_numpy(dtype=float)
        regime = out["regime_ma"].to_numpy(dtype=float)
        realized = out["matured_return"].to_numpy(dtype=float)
        total = np.zeros(len(data), dtype=float)
        payoff = np.zeros(len(data), dtype=float)
        wins = np.zeros(len(data), dtype=float)
        # Outcomes and query paths do not overlap; every outcome has already matured.
        for lag in range(params.shape_bars + params.horizon_bars,
                         min(params.history_bars, len(data) - 1) + 1):
            distance = np.mean((shape[lag:] - shape[:-lag]) ** 2, axis=1)
            outcome = realized[params.horizon_bars:len(data) - lag + params.horizon_bars]
            valid = (np.isfinite(distance) & np.isfinite(outcome)
                     & np.isfinite(regime[:-lag]) & (prices[:-lag] > regime[:-lag]))
            weight = np.zeros(len(distance), dtype=float)
            weight[valid] = np.exp(-distance[valid] / (2.0 * params.bandwidth ** 2))
            total[lag:] += weight
            payoff[lag:] += weight * np.where(valid, outcome, 0.0)
            wins[lag:] += weight * (outcome > 0.0)
        out["analog_support"] = total
        out["analog_forecast"] = np.divide(
            payoff, total, out=np.full(len(data), np.nan), where=total > 0.0)
        out["analog_hit_rate"] = np.divide(
            wins, total, out=np.full(len(data), np.nan), where=total > 0.0)
        return out.replace([np.inf, -np.inf], np.nan)

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame,
                         ctx: StrategyContext, params: AnalogParams) -> SignalFrame:
        z = indicators["z"]
        entries = ((data["close"] > indicators["regime_ma"])
                   & (z <= params.entry_z) & (z > z.shift(1))
                   & (indicators["analog_forecast"] >= params.min_forecast)
                   & (indicators["analog_hit_rate"] >= 0.55)
                   & (indicators["analog_support"] >= 3.0)).fillna(False).to_numpy()
        prices = data["close"].to_numpy(dtype=float)
        atr = indicators["atr"].to_numpy(dtype=float)
        regime = indicators["regime_ma"].to_numpy(dtype=float)
        signal = np.zeros(len(data), dtype=int)
        active = False
        age = 0
        high_water = -np.inf
        stop = -np.inf
        for i in range(len(data)):
            valid = (np.isfinite(prices[i]) and np.isfinite(atr[i])
                     and atr[i] > 0.0 and np.isfinite(regime[i]))
            if active:
                age += 1
                if valid:
                    # Begin tracking on the first holding bar after the entry decision.
                    high_water = max(high_water, prices[i])
                    stop = max(stop, high_water - params.atr_mult * atr[i])
                if (not valid or prices[i] <= stop or prices[i] <= regime[i]
                        or age >= params.horizon_bars):
                    active = False
                    continue
            elif valid and entries[i]:
                active = True
                age = 0
                high_water = -np.inf
                stop = -np.inf
            signal[i] = int(active)
        df = pd.DataFrame({"signal": signal, "size": 0.95}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check() -> None:
    params = AnalogParams()
    strategy = object.__new__(GeneratedStrategy)
    index = pd.date_range("2020-01-01", periods=7)
    data = pd.DataFrame({"close": [100., 103., 102., 99., 100., 101., 102.]}, index=index)
    indicators = pd.DataFrame(
        {"z": [-0.9, -0.8, -0.7, -0.6, -0.5, -0.4, -0.3],
         "regime_ma": 90., "atr": 1., "analog_forecast": 0.02,
         "analog_hit_rate": 0.7, "analog_support": 5.}, index=index)
    result = strategy.generate_signals(data, indicators, None, params).data
    assert result["signal"].tolist() == [0, 0, 1, 1, 0, 0, 0]
    assert (result["size"] > 0).all()
    index = pd.date_range("2018-01-01", periods=600)
    close = 100.0 + np.arange(600) * 0.03 + np.sin(np.arange(600) / 4.0)
    bars = pd.DataFrame({"open": close, "high": close + 1.,
                         "low": close - 1., "close": close, "volume": 1000.}, index=index)
    full = strategy.indicators(bars, params)
    prefix = strategy.indicators(bars.iloc[:520], params)
    pd.testing.assert_frame_equal(full.iloc[:520], prefix)
    assert strategy.warmup_bars(params) >= 452


if __name__ == "__main__":
    _self_check()
