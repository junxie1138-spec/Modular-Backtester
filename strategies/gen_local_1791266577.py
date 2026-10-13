from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class Params:
    ma_window: int = 60
    pressure_window: int = 5
    atr_window: int = 14
    entry_z: float = 1.5
    pressure_min: float = 0.15
    breakeven_pct: float = 0.025
    trail_atr: float = 3.0
    initial_stop_atr: float = 3.0
    max_hold: int = 18

    def __post_init__(self):
        if min(self.ma_window, self.pressure_window, self.atr_window) < 2 or self.max_hold < 1:
            raise ValueError("Lookbacks must be >= 2 and max_hold >= 1.")
        values = (self.entry_z, self.pressure_min, self.breakeven_pct,
                  self.trail_atr, self.initial_stop_atr)
        if not np.isfinite(values).all() or min(values) <= 0 or self.pressure_min > 1:
            raise ValueError("Thresholds must be finite and positive; pressure_min <= 1.")


class GeneratedStrategy(BaseStrategy[Params]):
    strategy_id = "gen_local_1791266577"

    @classmethod
    def params_type(cls):
        return Params

    def warmup_bars(self, params: Params) -> int:
        return int(max(params.ma_window, params.pressure_window, params.atr_window + 1))

    def indicators(self, data: pd.DataFrame, params: Params) -> pd.DataFrame:
        out = pd.DataFrame(index=data.index)
        close = data["close"]
        out["ma"] = close.rolling(params.ma_window, min_periods=params.ma_window).mean()
        out["std"] = close.rolling(params.ma_window, min_periods=params.ma_window).std(ddof=0)
        out["z"] = (close - out["ma"]) / out["std"].replace(0.0, np.nan)
        out["range"] = data["high"] - data["low"]
        out["body_fraction"] = ((close - data["open"]) / out["range"].replace(0.0, np.nan)).clip(-1.0, 1.0)
        out.loc[out["range"].eq(0.0) & close.eq(data["open"]), "body_fraction"] = 0.0
        out["body_volume"] = out["body_fraction"] * data["volume"]
        out["body_volume_sum"] = out["body_volume"].rolling(params.pressure_window, min_periods=params.pressure_window).sum()
        out["volume_sum"] = data["volume"].rolling(params.pressure_window, min_periods=params.pressure_window).sum()
        out["pressure"] = out["body_volume_sum"] / out["volume_sum"].replace(0.0, np.nan)
        previous_close = close.shift(1)
        out["true_range"] = pd.concat(
            [out["range"], (data["high"] - previous_close).abs(),
             (data["low"] - previous_close).abs()], axis=1
        ).max(axis=1)
        out.loc[previous_close.isna(), "true_range"] = np.nan
        out["atr"] = out["true_range"].rolling(params.atr_window, min_periods=params.atr_window).mean()
        return out.replace([np.inf, -np.inf], np.nan)

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame,
                         ctx: StrategyContext, params: Params) -> SignalFrame:
        n = len(data)
        desired = np.zeros(n, dtype=int)
        prices = data[["open", "high", "low", "close"]].to_numpy(dtype=float)
        z = indicators["z"].to_numpy(dtype=float)
        pressure = indicators["pressure"].to_numpy(dtype=float)
        atr = indicators["atr"].to_numpy(dtype=float)
        valid = (np.isfinite(prices).all(axis=1) & (prices > 0).all(axis=1)
                 & np.isfinite(z) & np.isfinite(pressure) & np.isfinite(atr) & (atr > 0))
        enter = valid & (z <= -params.entry_z) & (pressure >= params.pressure_min)
        pending = False
        active = False
        armed = False
        entry = peak = stop = 0.0
        age = 0
        for i in range(n):
            if pending:
                pending = False
                if not valid[i] or i == 0:
                    continue
                # The preceding close's decision fills at this bar's open.
                entry = prices[i, 0]
                peak = entry
                stop = entry - params.initial_stop_atr * atr[i - 1]
                age = 0
                armed = False
                active = True
            if active:
                if not valid[i]:
                    active = False
                    continue
                age += 1
                high, low, close = prices[i, 1:]
                # An observed breach requests a next-open exit, never an intrabar fill.
                if low <= stop or age >= params.max_hold:
                    active = False
                    continue
                peak = max(peak, high)
                if high >= entry * (1.0 + params.breakeven_pct):
                    armed = True
                if armed:
                    stop = max(stop, entry, peak - params.trail_atr * atr[i])
                    if close <= stop:
                        active = False
                        continue
                desired[i] = 1
            elif enter[i]:
                desired[i] = 1
                pending = True
        df = pd.DataFrame({"signal": desired, "size": np.full(n, 0.95, dtype=float)}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check():
    index = pd.date_range("2020-01-01", periods=8)
    data = pd.DataFrame({
        "open": [100, 100, 100, 101, 104, 104, 98, 98],
        "high": [101, 101, 101, 104, 106, 105, 99, 99],
        "low": [99, 99, 99, 100, 103, 102, 97, 97],
        "close": [100, 100, 100, 103, 105, 104, 98, 98],
        "volume": [1000] * 8,
    }, index=index)
    p = Params(ma_window=3, pressure_window=2, atr_window=2)
    features = pd.DataFrame({
        "z": [np.nan, np.nan, -2, 0, 0, 0, 0, 0],
        "pressure": [np.nan, np.nan, 0.4, 0, 0, 0, 0, 0],
        "atr": [1, 1, 1, 1, 1, 10, 1, 1],
    }, index=index)
    result = GeneratedStrategy.generate_signals(None, data, features, None, p).data
    assert result["signal"].tolist() == [0, 0, 0, 1, 1, 1, 0, 0]
    features.loc[index[2], "pressure"] = 0
    assert not GeneratedStrategy.generate_signals(None, data, features, None, p).data["signal"].any()
    features.loc[index[2], "pressure"] = 0.4
    p.max_hold = 2
    assert GeneratedStrategy.generate_signals(None, data, features, None, p).data["signal"].tolist() == [0, 0, 0, 1, 1, 0, 0, 0]
    actual = GeneratedStrategy.indicators(None, data, p)
    assert actual.index.equals(data.index)
    assert GeneratedStrategy.warmup_bars(None, p) >= 3


if __name__ == "__main__":
    _self_check()
