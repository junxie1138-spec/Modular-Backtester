from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class StrategyParams:
    fractional_order: float = 0.40
    fractional_window: int = 32
    efficiency_window: int = 6
    baseline_window: int = 20
    volatility_window: int = 20
    atr_window: int = 14
    efficiency_threshold: float = 0.55
    target_volatility: float = 0.12
    breakeven_gain: float = 0.012
    atr_multiple: float = 2.5
    max_holding_bars: int = 10
    spike_multiple: float = 2.5
    refractory_bars: int = 3


class GeneratedStrategy(BaseStrategy[StrategyParams]):
    strategy_id = "gen_local-w2_1791304497"

    @classmethod
    def params_type(cls) -> type[StrategyParams]:
        return StrategyParams

    def warmup_bars(self, params: StrategyParams) -> int:
        return int(params.fractional_window + max(
            params.efficiency_window, params.baseline_window,
            params.volatility_window + params.refractory_bars + 1,
            params.atr_window
        ) + 2)

    def indicators(self, data: pd.DataFrame, params: StrategyParams) -> pd.DataFrame:
        windows = (params.fractional_window, params.efficiency_window,
                   params.baseline_window, params.volatility_window,
                   params.atr_window)
        if any(window < 2 for window in windows):
            raise ValueError("Indicator windows must be at least two bars")
        if not 0.0 < params.fractional_order < 1.0:
            raise ValueError("fractional_order must lie between zero and one")
        if not 0.0 < params.efficiency_threshold <= 1.0:
            raise ValueError("Invalid efficiency threshold")
        if (params.target_volatility <= 0 or params.breakeven_gain <= 0
                or params.atr_multiple <= 0 or params.spike_multiple <= 0
                or params.max_holding_bars < 1 or params.refractory_bars < 0):
            raise ValueError("Invalid risk or holding parameters")

        out = pd.DataFrame(index=data.index)
        close = data["close"].astype(float).where(data["close"] > 0)
        out["log_close"] = np.log(close)
        out["log_return"] = out["log_close"].diff()
        out["fractional_close"] = out["log_close"].copy()
        weight = 1.0
        for lag in range(1, params.fractional_window):
            weight *= (lag - 1 - params.fractional_order) / lag
            out["fractional_close"] += weight * out["log_close"].shift(lag)

        out["fractional_change"] = out["fractional_close"].diff()
        out["fractional_displacement"] = out["fractional_change"].rolling(
            params.efficiency_window, min_periods=params.efficiency_window
        ).sum()
        out["fractional_path"] = out["fractional_change"].abs().rolling(
            params.efficiency_window, min_periods=params.efficiency_window
        ).sum()
        out["efficiency"] = out["fractional_displacement"] / out[
            "fractional_path"
        ].replace(0.0, np.nan)
        out["fractional_baseline"] = out["fractional_close"].rolling(
            params.baseline_window, min_periods=params.baseline_window
        ).mean()
        out["daily_volatility"] = out["log_return"].rolling(
            params.volatility_window, min_periods=params.volatility_window
        ).std(ddof=0)
        out["annual_volatility"] = out["daily_volatility"] * np.sqrt(252.0)
        out["previous_close"] = close.shift(1)
        out["true_range"] = pd.concat([
            data["high"] - data["low"],
            (data["high"] - out["previous_close"]).abs(),
            (data["low"] - out["previous_close"]).abs()
        ], axis=1).max(axis=1)
        out["atr"] = out["true_range"].rolling(
            params.atr_window, min_periods=params.atr_window
        ).mean()
        out["spike"] = (
            out["log_return"].abs()
            > params.spike_multiple * out["daily_volatility"].shift(1)
        ).fillna(False)
        out["cooldown"] = out["spike"].astype(float).rolling(
            params.refractory_bars + 1, min_periods=1
        ).max().gt(0.0)
        out["target_size"] = (
            params.target_volatility
            / out["annual_volatility"].where(out["annual_volatility"] > 1e-8)
        ).clip(0.05, 0.95).fillna(0.05)
        out["eligible"] = (
            (out["efficiency"] >= params.efficiency_threshold)
            & (out["fractional_change"] > 0.0)
            & (out["fractional_close"] > out["fractional_baseline"])
            & (out["daily_volatility"] > 1e-8)
            & (out["atr"] > 0.0)
            & ~out["cooldown"]
        ).fillna(False)
        out["confirmed"] = (
            out["eligible"] & out["eligible"].shift(1, fill_value=False)
        ).fillna(False)
        return out

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame,
                         ctx: StrategyContext, params: StrategyParams) -> SignalFrame:
        count = len(data)
        signals = np.zeros(count, dtype=int)
        opens = data["open"].to_numpy(dtype=float)
        closes = data["close"].to_numpy(dtype=float)
        atrs = indicators["atr"].to_numpy(dtype=float)
        confirmed = indicators["confirmed"].to_numpy(dtype=bool)
        active = False
        pending_entry = False
        armed = False
        entry_price = 0.0
        high_water = 0.0
        stop = 0.0
        age = 0

        for i in range(count):
            if pending_entry:
                pending_entry = False
                previous_atr = atrs[i - 1] if i > 0 else np.nan
                if (np.isfinite(opens[i]) and opens[i] > 0.0
                        and np.isfinite(previous_atr) and previous_atr > 0.0):
                    active = True
                    armed = False
                    age = 0
                    entry_price = opens[i]
                    high_water = entry_price
                    stop = entry_price - params.atr_multiple * previous_atr

            if active:
                age += 1
                close = closes[i]
                valid_close = np.isfinite(close) and close > 0.0
                exit_now = age >= params.max_holding_bars or not valid_close
                if valid_close:
                    high_water = max(high_water, close)
                    if close >= entry_price * (1.0 + params.breakeven_gain):
                        armed = True
                    if armed:
                        stop = max(stop, entry_price)
                        if np.isfinite(atrs[i]) and atrs[i] > 0.0:
                            stop = max(stop, high_water - params.atr_multiple * atrs[i])
                    exit_now = exit_now or close <= stop
                if exit_now:
                    active = False
                else:
                    signals[i] = 1
            elif (confirmed[i] and np.isfinite(closes[i]) and closes[i] > 0.0
                  and np.isfinite(atrs[i]) and atrs[i] > 0.0):
                signals[i] = 1
                pending_entry = True

        df = pd.DataFrame(index=data.index)
        df["signal"] = signals
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = indicators["target_size"].shift(1).fillna(0.05).astype(float)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check() -> None:
    strategy = GeneratedStrategy()
    params = StrategyParams(max_holding_bars=4)
    index = pd.date_range("2020-01-01", periods=12, freq="B")
    close = np.array([100, 100, 100, 102, 104, 103, 101, 101, 101, 101, 101, 101], dtype=float)
    data = pd.DataFrame({"open": 100.0, "high": close + 1.0,
                         "low": close - 1.0, "close": close,
                         "volume": 1000.0}, index=index)
    ind = pd.DataFrame({"atr": 1.0, "target_size": 0.5,
                        "confirmed": False}, index=index)
    ind.loc[index[1:3], "confirmed"] = True
    result = strategy.generate_signals(data, ind, None, params).data
    assert result["signal"].tolist() == [0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0]
    assert result["size"].gt(0).all()
    altered = data.copy()
    altered.loc[index[8:], "close"] = 50.0
    prefix = strategy.generate_signals(altered, ind, None, params).data.iloc[:8]
    pd.testing.assert_frame_equal(result.iloc[:8], prefix)
    calculated = strategy.indicators(data, StrategyParams())
    assert not calculated["confirmed"].any()
    assert calculated["target_size"].gt(0).all()


if __name__ == "__main__":
    _self_check()
