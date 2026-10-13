from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class AnalogParams:
    roc_bars: int = 3
    volatility_bars: int = 20
    shape_bars: int = 5
    history_bars: int = 504
    neighbors: int = 12
    entry_strength: float = 0.40
    max_distance: float = 1.25
    profit_target: float = 0.025
    hold_bars: int = 8
    max_size: float = 1.0


class GeneratedStrategy(BaseStrategy[AnalogParams]):
    strategy_id = "gen_local_1790941127"

    @classmethod
    def params_type(cls):
        return AnalogParams

    def warmup_bars(self, params: AnalogParams) -> int:
        return int(params.history_bars + params.hold_bars + params.shape_bars
                   + max(params.volatility_bars + 1, params.roc_bars + 2))

    def indicators(self, data: pd.DataFrame, params: AnalogParams) -> pd.DataFrame:
        if (min(params.roc_bars, params.shape_bars, params.history_bars,
                params.neighbors, params.hold_bars) < 1
                or params.volatility_bars < 2
                or params.neighbors > params.history_bars
                or not 0 < params.entry_strength < 1
                or not 0 < params.max_size <= 1
                or params.max_distance <= 0 or params.profit_target <= 0):
            raise ValueError("Invalid analog parameters")
        out = pd.DataFrame(index=data.index)
        close = data["close"].where(data["close"] > 0)
        opening = data["open"].where(data["open"] > 0)
        out["return"] = close.pct_change(fill_method=None)
        out["volatility"] = out["return"].rolling(params.volatility_bars).std()
        out["roc"] = close.pct_change(params.roc_bars, fill_method=None)
        out["acceleration"] = out["roc"].diff() / out["volatility"].replace(0, np.nan)
        shape_columns = []
        for lag in range(params.shape_bars):
            name = f"acceleration_lag_{lag}"
            out[name] = out["acceleration"].shift(lag)
            shape_columns.append(name)
        # Endpoint t records an open-to-close outcome that has already completed.
        out["matured_return"] = close / opening.shift(params.hold_bars - 1) - 1
        out = out.replace([np.inf, -np.inf], np.nan)
        features = out[shape_columns].to_numpy(dtype=float)
        outcomes = out["matured_return"].to_numpy(dtype=float)
        volatility = out["volatility"].to_numpy(dtype=float)
        forecast = np.zeros(len(data), dtype=float)
        strength = np.zeros(len(data), dtype=float)
        distance = np.full(len(data), np.nan)
        # ponytail: O(bars * history * shape); index the shapes if histories grow substantially.
        for i in range(self.warmup_bars(params) - 1, len(data)):
            if not np.isfinite(features[i]).all():
                continue
            last = i - max(params.hold_bars, params.shape_bars)
            candidates = np.arange(last - params.history_bars + 1, last + 1)
            endpoints = candidates + params.hold_bars
            valid = (np.isfinite(features[candidates]).all(axis=1)
                     & np.isfinite(outcomes[endpoints])
                     & np.isfinite(volatility[candidates])
                     & (volatility[candidates] > 0))
            candidates = candidates[valid]
            if len(candidates) < params.neighbors:
                continue
            distances = np.sqrt(np.mean((features[candidates] - features[i]) ** 2, axis=1))
            chosen = np.argpartition(distances, params.neighbors - 1)[:params.neighbors]
            selected = candidates[chosen]
            selected_distances = distances[chosen]
            distance[i] = float(selected_distances.mean())
            if distance[i] > params.max_distance:
                continue
            weights = 1.0 / (0.10 + selected_distances)
            labels = (outcomes[selected + params.hold_bars]
                      / (volatility[selected] * np.sqrt(params.hold_bars)))
            mean = float(np.average(labels, weights=weights))
            rms = float(np.sqrt(np.average(labels ** 2, weights=weights)))
            forecast[i] = mean
            strength[i] = float(np.clip(abs(mean) / max(rms, 1e-12), 0, 1))
        out["analog_distance"] = distance
        out["forecast"] = forecast
        out["strength"] = strength
        return out

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame,
                         ctx: StrategyContext, params: AnalogParams) -> SignalFrame:
        n = len(data)
        signals = np.zeros(n, dtype=int)
        sizes = np.full(n, params.max_size * 0.25, dtype=float)
        opening = data["open"].to_numpy(dtype=float)
        close = data["close"].to_numpy(dtype=float)
        forecast = indicators["forecast"].to_numpy(dtype=float)
        strength = indicators["strength"].to_numpy(dtype=float)
        side = 0
        entry_bar = -1
        entry_price = np.nan
        active_size = params.max_size * 0.25
        for i in range(n):
            if side:
                # Read the fill bar's open only once that bar has occurred.
                if i == entry_bar:
                    entry_price = opening[i]
                valid_price = np.isfinite(entry_price) and entry_price > 0
                gain = side * (close[i] / entry_price - 1) if valid_price else np.nan
                if (not valid_price or not np.isfinite(close[i])
                        or gain >= params.profit_target
                        or i - entry_bar + 1 >= params.hold_bars):
                    side = 0
                    signals[i] = 0
                    continue
            elif (i >= self.warmup_bars(params) - 1
                  and np.isfinite(close[i]) and close[i] > 0
                  and np.isfinite(forecast[i]) and forecast[i] != 0
                  and np.isfinite(strength[i]) and strength[i] >= params.entry_strength):
                side = 1 if forecast[i] > 0 else -1
                entry_bar = i + 1
                active_size = params.max_size * (0.25 + 0.75 * np.clip(strength[i], 0, 1))
            signals[i] = side
            if side:
                sizes[i] = active_size
        df = pd.DataFrame({"signal": signals, "size": sizes}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = df["size"].shift(1).fillna(params.max_size * 0.25).astype(float)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check():
    params = AnalogParams(roc_bars=2, volatility_bars=5, shape_bars=3,
                          history_bars=30, neighbors=3, hold_bars=6)
    strategy = GeneratedStrategy()
    n = 120
    index = pd.date_range("2020-01-01", periods=n, freq="B")
    prices = 100 * np.exp(np.cumsum(0.003 * np.sin(np.arange(n) * 0.7)))
    data = pd.DataFrame({"open": prices, "high": prices * 1.01,
                         "low": prices * 0.99, "close": prices, "volume": 1000.0}, index=index)
    full = strategy.indicators(data, params)
    prefix = strategy.indicators(data.iloc[:90], params)
    assert full.iloc[:90].equals(prefix)
    ctx = StrategyContext(symbol="SPY", timeframe="1d", warmup_bars=strategy.warmup_bars(params))
    decision = strategy.warmup_bars(params) - 1
    data.loc[:, ["open", "close"]] = 100.0
    forced = pd.DataFrame({"forecast": 0.0, "strength": 0.0}, index=index)
    forced.iloc[decision] = [1.0, 0.8]
    result = strategy.generate_signals(data, forced, ctx, params).data
    assert result["signal"].iloc[decision] == 0
    assert (result["signal"].iloc[decision + 1:decision + params.hold_bars + 1] == 1).all()
    assert result["signal"].iloc[decision + params.hold_bars + 1] == 0
    assert np.isclose(result["size"].iloc[decision + 1], 0.85)
    data.iloc[decision + 2, data.columns.get_loc("close")] = 103.0
    result = strategy.generate_signals(data, forced, ctx, params).data
    assert result["signal"].iloc[decision + 2] == 1
    assert result["signal"].iloc[decision + 3] == 0
    forced.iloc[decision] = [-1.0, 0.8]
    data.iloc[decision + 2, data.columns.get_loc("close")] = 97.0
    result = strategy.generate_signals(data, forced, ctx, params).data
    assert result["signal"].iloc[decision + 2] == -1
    assert result["signal"].iloc[decision + 3] == 0
    assert result["signal"].isin([-1, 0, 1]).all()
    assert np.isfinite(result["size"]).all() and (result["size"] > 0).all()


if __name__ == "__main__":
    _self_check()
