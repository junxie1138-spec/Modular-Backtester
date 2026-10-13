from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class AccelerationParams:
    fast_window: int = 5
    medium_window: int = 15
    volatility_window: int = 20
    atr_window: int = 14
    acceleration_threshold: float = 0.45
    target_volatility: float = 0.12
    trailing_atr: float = 3.0
    max_hold_bars: int = 18


class GeneratedStrategy(BaseStrategy[AccelerationParams]):
    strategy_id = "gen_local-w2_1791328063"

    @classmethod
    def params_type(cls) -> type[AccelerationParams]:
        return AccelerationParams

    def warmup_bars(self, params: AccelerationParams) -> int:
        return int(max(200, 2 * params.fast_window + 1,
                       params.medium_window + 1,
                       params.volatility_window + 1,
                       params.atr_window + 1))

    def indicators(self, data: pd.DataFrame,
                   params: AccelerationParams) -> pd.DataFrame:
        if min(params.fast_window, params.medium_window,
               params.volatility_window, params.atr_window,
               params.max_hold_bars) < 1:
            raise ValueError("Windows and holding period must be positive")
        if not all(np.isfinite(value) and value > 0 for value in
                   (params.acceleration_threshold, params.target_volatility,
                    params.trailing_atr)):
            raise ValueError("Threshold, volatility target and ATR multiple must be positive")
        close = data["close"].where(data["close"] > 0)
        out = pd.DataFrame(index=data.index)
        out["return"] = close.pct_change(fill_method=None)
        out["daily_volatility"] = out["return"].rolling(
            params.volatility_window,
            min_periods=params.volatility_window).std(ddof=0)
        out["annual_volatility"] = out["daily_volatility"] * np.sqrt(252.0)
        out["fast_roc"] = close.pct_change(
            periods=params.fast_window, fill_method=None)
        out["medium_roc"] = close.pct_change(
            periods=params.medium_window, fill_method=None)
        out["roc_acceleration"] = out["fast_roc"].diff(params.fast_window)
        denominator = out["daily_volatility"] * np.sqrt(float(params.fast_window))
        out["normalized_acceleration"] = out["roc_acceleration"].div(
            denominator.where(denominator > 0))
        previous_close = close.shift(1)
        out["true_range"] = pd.concat([
            data["high"] - data["low"],
            (data["high"] - previous_close).abs(),
            (data["low"] - previous_close).abs()
        ], axis=1).max(axis=1, skipna=False)
        out["atr"] = out["true_range"].rolling(
            params.atr_window, min_periods=params.atr_window).mean()
        out["ma200"] = close.rolling(200, min_periods=200).mean()
        acceleration = out["normalized_acceleration"]
        out["entry"] = (
            (acceleration > params.acceleration_threshold)
            & (acceleration.shift(1) <= params.acceleration_threshold)
            & (out["fast_roc"] > 0)
            & (out["medium_roc"] < 0)
            & (close > out["ma200"])
        ).fillna(False)
        return out.replace([np.inf, -np.inf], np.nan)

    def generate_signals(self, data: pd.DataFrame,
                         indicators: pd.DataFrame,
                         ctx: StrategyContext,
                         params: AccelerationParams) -> SignalFrame:
        df = pd.DataFrame(index=data.index)
        signals = np.zeros(len(data), dtype=int)
        closes = data["close"].to_numpy(dtype=float)
        atrs = indicators["atr"].to_numpy(dtype=float)
        averages = indicators["ma200"].to_numpy(dtype=float)
        volatilities = indicators["annual_volatility"].to_numpy(dtype=float)
        entries = indicators["entry"].fillna(False).to_numpy(dtype=bool)
        active = False
        age = 0
        high_water = 0.0
        stop = 0.0
        for i in range(len(data)):
            price, atr, average, volatility = (
                closes[i], atrs[i], averages[i], volatilities[i])
            valid = (np.isfinite(price) and price > 0
                     and np.isfinite(atr) and atr > 0
                     and np.isfinite(average)
                     and np.isfinite(volatility) and volatility > 0)
            if active:
                age += 1
                if valid:
                    high_water = max(high_water, price)
                    stop = max(stop, high_water - params.trailing_atr * atr)
                if (not valid or price <= stop or price <= average
                        or age >= params.max_hold_bars):
                    active = False
            elif valid and entries[i] and price > average:
                active = True
                age = 0
                high_water = price
                stop = price - params.trailing_atr * atr
            signals[i] = int(active)
        df["signal"] = signals
        df["size"] = (params.target_volatility / indicators[
            "annual_volatility"].where(indicators["annual_volatility"] > 0)
        ).replace([np.inf, -np.inf], np.nan).clip(0.05, 0.95)
        df["size"] = df["size"].shift(1).fillna(0.05).astype(float)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


if __name__ == "__main__":
    index = pd.date_range("2020-01-01", periods=6)
    prices = np.array([100.0, 110.0, 109.0, 108.0, 100.0, 101.0])
    data = pd.DataFrame({"open": prices, "high": prices + 1,
                         "low": prices - 1, "close": prices,
                         "volume": 1000.0}, index=index)
    indicators = pd.DataFrame({"atr": [2., 2., 10., 10., 2., 2.],
                               "ma200": 90.0, "annual_volatility": 0.2,
                               "entry": [True, False, False, False, False, True]},
                              index=index)
    strategy = GeneratedStrategy()
    result = strategy.generate_signals(data, indicators, None, AccelerationParams()).data
    assert result["signal"].tolist() == [0, 1, 1, 1, 1, 0]
    assert (result["size"] > 0).all()
    result = strategy.generate_signals(
        data, indicators, None, AccelerationParams(max_hold_bars=2)).data
    assert result["signal"].tolist() == [0, 1, 1, 0, 0, 0]
    assert strategy.warmup_bars(AccelerationParams()) >= 200
