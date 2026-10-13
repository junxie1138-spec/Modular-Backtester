from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class AnalogParams:
    ma_window: int = 20
    history_bars: int = 252
    neighbors: int = 8
    max_distance: float = 0.65
    z_entry: float = 1.0
    forecast_min: float = 0.002
    pressure_min: float = 0.10
    profit_target: float = 0.012
    hold_bars: int = 4


class GeneratedStrategy(BaseStrategy[AnalogParams]):
    strategy_id = "gen_local_1790882821"

    @classmethod
    def params_type(cls) -> type[AnalogParams]:
        return AnalogParams

    @staticmethod
    def warmup_bars(params: AnalogParams) -> int:
        return int(params.ma_window + params.history_bars + params.hold_bars + 4)

    @staticmethod
    def indicators(data: pd.DataFrame, params: AnalogParams) -> pd.DataFrame:
        if params.ma_window < 2 or params.neighbors < 1 or params.history_bars < params.neighbors:
            raise ValueError("Invalid analog lookbacks")
        if params.max_distance <= 0 or not 3 <= params.hold_bars <= 5:
            raise ValueError("Invalid analog distance or holding horizon")
        out = pd.DataFrame(index=data.index)
        close = data["close"].astype(float)
        opening = data["open"].astype(float)
        out["ma"] = close.rolling(params.ma_window).mean()
        out["std"] = close.rolling(params.ma_window).std(ddof=0)
        out["z"] = ((close - out["ma"]) / out["std"].replace(0, np.nan)).replace([np.inf, -np.inf], np.nan)
        out["z_lag1"] = out["z"].shift(1)
        out["z_lag3"] = out["z"].shift(3)
        span = (data["high"] - data["low"]).where(lambda x: x > 0)
        out["body_pressure"] = ((close - opening) / span).replace([np.inf, -np.inf], np.nan)
        out["pressure"] = out["body_pressure"].rolling(2).mean()
        features = out[["z", "z_lag1", "z_lag3"]].to_numpy(dtype=float)
        prices = close.to_numpy()
        opens = opening.to_numpy()
        forecast = np.full(len(data), np.nan)
        distance = np.full(len(data), np.nan)
        first = params.ma_window + 2
        # ponytail: bounded O(bars * history) matching; index neighbors if histories become large.
        for t in range(first, len(data)):
            if not np.isfinite(features[t]).all():
                continue
            start = max(first, t - params.history_bars)
            # Historical outcomes end before the current three-bar trajectory starts.
            stop = t - params.hold_bars - 3
            if stop <= start:
                continue
            candidates = np.arange(start, stop)
            errors = np.sqrt(np.mean((features[candidates] - features[t]) ** 2, axis=1))
            entry = opens[candidates + 1]
            terminal = prices[candidates + params.hold_bars]
            valid = np.isfinite(errors) & (errors <= params.max_distance)
            valid &= np.isfinite(entry) & (entry > 0) & np.isfinite(terminal) & (terminal > 0)
            candidates = candidates[valid]
            errors = errors[valid]
            if len(candidates) < params.neighbors:
                continue
            nearest = np.argpartition(errors, params.neighbors - 1)[:params.neighbors]
            matched = candidates[nearest]
            forecast[t] = np.mean(prices[matched + params.hold_bars] / opens[matched + 1] - 1.0)
            distance[t] = np.mean(errors[nearest])
        out["analog_forecast"] = forecast
        out["analog_distance"] = distance
        return out

    @staticmethod
    def generate_signals(data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: AnalogParams) -> SignalFrame:
        if params.profit_target <= 0 or not 3 <= params.hold_bars <= 5:
            raise ValueError("Profit target must be positive and holding horizon must be 3-5 bars")
        analog_agrees = (indicators["z"] <= -params.z_entry) & (indicators["analog_forecast"] >= params.forecast_min)
        pressure_agrees = indicators["pressure"] >= params.pressure_min
        eligible = (analog_agrees & pressure_agrees).fillna(False).to_numpy(dtype=bool)
        close = data["close"].to_numpy(dtype=float)
        opening = data["open"].to_numpy(dtype=float)
        decisions = np.zeros(len(data), dtype=int)
        active = False
        entry_bar = -1
        entry_price = np.nan
        for i in range(len(data)):
            if active:
                if i == entry_bar:
                    entry_price = opening[i]
                if not np.isfinite(entry_price) or entry_price <= 0:
                    active = False
                    continue
                target_hit = np.isfinite(close[i]) and close[i] >= entry_price * (1.0 + params.profit_target)
                time_stop = i - entry_bar + 1 >= params.hold_bars
                if target_hit or time_stop:
                    active = False
                else:
                    decisions[i] = 1
            elif eligible[i] and np.isfinite(close[i]) and close[i] > 0:
                decisions[i] = 1
                active = True
                entry_bar = i + 1
                entry_price = np.nan
        df = pd.DataFrame({"signal": decisions, "size": 1.0}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check() -> None:
    index = pd.date_range("2020-01-01", periods=7)
    data = pd.DataFrame({"open": 100.0, "high": 103.0, "low": 99.0, "close": 100.0, "volume": 1000.0}, index=index)
    indicators = pd.DataFrame({"z": [-2.0] + [0.0] * 6, "analog_forecast": 0.01, "pressure": 0.5}, index=index)
    params = AnalogParams(profit_target=0.01)
    data.loc[index[2], "close"] = 102.0
    target = GeneratedStrategy.generate_signals(data, indicators, None, params).data
    assert target["signal"].tolist() == [0, 1, 1, 0, 0, 0, 0]
    data["close"] = 100.0
    expiry = GeneratedStrategy.generate_signals(data, indicators, None, params).data
    assert expiry["signal"].tolist() == [0, 1, 1, 1, 1, 0, 0]
    indicators["pressure"] = -0.5
    rejected = GeneratedStrategy.generate_signals(data, indicators, None, params).data
    assert rejected["signal"].eq(0).all()
    assert expiry["size"].gt(0).all()


if __name__ == "__main__":
    _self_check()
