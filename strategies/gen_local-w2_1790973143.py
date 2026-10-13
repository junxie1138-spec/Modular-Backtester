from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class AnalogParams:
    ma_days: int = 20
    shape_days: int = 5
    history_days: int = 252
    neighbors: int = 7
    hold_days: int = 4
    atr_days: int = 14
    z_floor: float = 0.75
    forecast_floor: float = 0.003
    pressure_floor: float = 0.10
    breakeven_pct: float = 0.008
    trail_atr: float = 1.5


class GeneratedStrategy(BaseStrategy[AnalogParams]):
    strategy_id = "gen_local-w2_1790973143"

    @classmethod
    def params_type(cls) -> type[AnalogParams]:
        return AnalogParams

    def warmup_bars(self, params: AnalogParams) -> int:
        return int(params.ma_days + params.shape_days + params.history_days
                   + params.hold_days + params.atr_days + 1)

    def indicators(self, data: pd.DataFrame, params: AnalogParams) -> pd.DataFrame:
        if (params.ma_days < 2 or params.shape_days < 2
                or params.history_days < params.neighbors or params.neighbors < 1
                or params.atr_days < 1 or not 3 <= params.hold_days <= 5
                or min(params.z_floor, params.forecast_floor, params.pressure_floor,
                       params.breakeven_pct, params.trail_atr) <= 0):
            raise ValueError("Invalid strategy parameters")
        close = data["close"].astype(float)
        out = pd.DataFrame(index=data.index)
        out["ma"] = close.rolling(params.ma_days).mean()
        out["std"] = close.rolling(params.ma_days).std(ddof=0)
        out["z"] = (close - out["ma"]) / out["std"].replace(0.0, np.nan)
        previous = close.shift(1)
        out["tr"] = pd.concat([
            data["high"] - data["low"],
            (data["high"] - previous).abs(),
            (data["low"] - previous).abs(),
        ], axis=1).max(axis=1)
        out["atr"] = out["tr"].rolling(params.atr_days).mean()
        spread = (data["high"] - data["low"]).replace(0.0, np.nan)
        out["body"] = ((close - data["open"]) / spread).clip(-1.0, 1.0).fillna(0.0)
        volume = data["volume"].clip(lower=0.0)
        out["pressure"] = (out["body"] * volume).rolling(3).sum() / volume.rolling(3).sum().replace(0.0, np.nan)
        n = len(data)
        forecast = np.full(n, np.nan)
        match_distance = np.full(n, np.nan)
        count = np.zeros(n, dtype=int)
        if n:
            distances = []
            outcomes = []
            # Each historical endpoint and its entire future outcome precede today's close.
            for lag in range(params.hold_days + 1,
                             params.hold_days + params.history_days + 1):
                distance = pd.Series(0.0, index=data.index)
                for offset in range(params.shape_days):
                    distance = distance + (out["z"].shift(offset)
                                           - out["z"].shift(lag + offset)) ** 2
                distances.append((distance / params.shape_days).to_numpy())
                outcomes.append((close.shift(lag - params.hold_days)
                                 / close.shift(lag) - 1.0).to_numpy())
            # ponytail: overlapping daily analogs; use episode sampling if dependence dominates.
            d = np.column_stack(distances)
            y = np.column_stack(outcomes)
            valid = np.isfinite(d) & np.isfinite(y)
            d = np.where(valid, d, np.inf)
            k = params.neighbors
            nearest = np.argpartition(d, k - 1, axis=1)[:, :k]
            selected_d = np.take_along_axis(d, nearest, axis=1)
            selected_y = np.take_along_axis(y, nearest, axis=1)
            usable = np.isfinite(selected_d)
            count = usable.sum(axis=1)
            ready = count == k
            forecast[ready] = np.where(usable, selected_y, 0.0).sum(axis=1)[ready] / k
            match_distance[ready] = selected_d[ready].mean(axis=1)
        out["analog_return"] = forecast
        out["analog_distance"] = match_distance
        out["analog_count"] = count
        return out.replace([np.inf, -np.inf], np.nan)

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame,
                         ctx: StrategyContext, params: AnalogParams) -> SignalFrame:
        z = indicators["z"]
        forecast = indicators["analog_return"]
        pressure = indicators["pressure"]
        # Primitive one: extreme z-score shapes whose historical outcomes predict recovery.
        # Primitive two: actual volume-weighted intraday pressure agrees with that recovery.
        long_entry = ((z <= -params.z_floor) & (forecast >= params.forecast_floor)
                      & (pressure >= params.pressure_floor))
        short_entry = ((z >= params.z_floor) & (forecast <= -params.forecast_floor)
                       & (pressure <= -params.pressure_floor))
        raw = np.zeros(len(data), dtype=int)
        side = 0
        entry = np.nan
        stop = -np.inf
        armed = False
        age = 0
        opens = data["open"].to_numpy(dtype=float)
        closes = data["close"].to_numpy(dtype=float)
        atr = indicators["atr"].to_numpy(dtype=float)
        longs = long_entry.to_numpy()
        shorts = short_entry.to_numpy()
        for i in range(len(data)):
            if side:
                # Yesterday's decision fills at today's open, never yesterday's close.
                if np.isnan(entry):
                    entry = opens[i]
                    if not np.isfinite(entry) or entry <= 0:
                        side = 0
                        raw[i] = 0
                        continue
                    stop = side * entry - params.trail_atr * atr[i - 1]
                age += 1
                if not np.isfinite(closes[i]) or not np.isfinite(atr[i]) or atr[i] <= 0:
                    side = 0
                else:
                    signed_close = side * closes[i]
                    breached = signed_close <= stop
                    if side * (closes[i] / entry - 1.0) >= params.breakeven_pct:
                        armed = True
                    if armed:
                        # Signed stops only rise; the corresponding short price stop only falls.
                        stop = max(stop, side * entry,
                                   signed_close - params.trail_atr * atr[i])
                    if breached or signed_close <= stop or age >= params.hold_days:
                        side = 0
                raw[i] = side
                continue
            if np.isfinite(atr[i]) and atr[i] > 0:
                side = 1 if longs[i] else (-1 if shorts[i] else 0)
            if side:
                entry = np.nan
                armed = False
                age = 0
            raw[i] = side
        df = pd.DataFrame({"signal": raw, "size": 0.95}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check() -> None:
    index = pd.date_range("2020-01-01", periods=10)
    data = pd.DataFrame({"open": 100.0, "high": 103.0, "low": 97.0,
                         "close": [100, 102, 100, 100, 100, 100, 98, 100, 100, 100],
                         "volume": 1000.0}, index=index)
    features = pd.DataFrame({"z": 0.0, "analog_return": 0.0,
                             "pressure": 0.0, "atr": 1.0}, index=index)
    features.loc[index[0], ["z", "analog_return", "pressure"]] = [-2.0, 0.02, 0.5]
    features.loc[index[5], ["z", "analog_return", "pressure"]] = [2.0, -0.02, -0.5]
    result = GeneratedStrategy().generate_signals(data, features, None, AnalogParams())
    assert result.data["signal"].tolist() == [0, 1, 1, 0, 0, 0, -1, -1, 0, 0]
    assert (result.data["size"] > 0).all()


if __name__ == "__main__":
    _self_check()
