from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class RegimeAuctionParams:
    ma_bars: int = 40
    regime_bars: int = 20
    pressure_bars: int = 5
    z_entry: float = 1.0
    pressure_entry: float = 0.15
    trend_efficiency: float = 0.35
    range_efficiency: float = 0.20
    profit_target: float = 0.04
    max_hold_bars: int = 18

    def __post_init__(self):
        if min(self.ma_bars, self.regime_bars, self.pressure_bars) < 2:
            raise ValueError("Lookbacks must be at least two bars")
        if self.max_hold_bars < 1 or self.profit_target <= 0:
            raise ValueError("Exit parameters must be positive")
        if self.z_entry <= 0 or not 0 < self.pressure_entry < 1:
            raise ValueError("Invalid entry thresholds")
        if not 0 <= self.range_efficiency < self.trend_efficiency <= 1:
            raise ValueError("Invalid regime thresholds")


class GeneratedStrategy(BaseStrategy[RegimeAuctionParams]):
    strategy_id = "gen_local-w2_1791324599"

    @classmethod
    def params_type(cls):
        return RegimeAuctionParams

    def warmup_bars(self, params):
        return int(max(params.ma_bars, params.regime_bars + 1, params.pressure_bars))

    def indicators(self, data, params):
        close = data["close"].astype(float)
        mean = close.rolling(params.ma_bars, min_periods=params.ma_bars).mean()
        deviation = close.rolling(params.ma_bars, min_periods=params.ma_bars).std(ddof=0)
        change = close.diff()
        path = change.abs().rolling(params.regime_bars, min_periods=params.regime_bars).sum()
        displacement = close.diff(params.regime_bars)
        span = (data["high"] - data["low"]).replace(0, np.nan)
        location = ((2 * close - data["high"] - data["low"]) / span).clip(-1, 1)
        pressure = location.rolling(params.pressure_bars, min_periods=params.pressure_bars).mean()
        return pd.DataFrame({
            "mean": mean,
            "deviation": deviation,
            "change": change,
            "path": path,
            "displacement": displacement,
            "location": location,
            "pressure": pressure,
            "z": (close - mean) / deviation.replace(0, np.nan),
            "efficiency": displacement.abs() / path.replace(0, np.nan),
        }, index=data.index).replace([np.inf, -np.inf], np.nan)

    def generate_signals(self, data, indicators, ctx, params):
        z = indicators["z"]
        pressure = indicators["pressure"]
        efficiency = indicators["efficiency"]
        trend = efficiency >= params.trend_efficiency
        ranging = efficiency <= params.range_efficiency
        long_entry = (pressure >= params.pressure_entry) & (
            (trend & (z >= params.z_entry)) | (ranging & (z <= -params.z_entry))
        )
        short_entry = (pressure <= -params.pressure_entry) & (
            (trend & (z <= -params.z_entry)) | (ranging & (z >= params.z_entry))
        )
        ready = indicators[["z", "pressure", "efficiency"]].notna().all(axis=1)
        entries = np.select(
            [ready & long_entry, ready & short_entry], [1, -1], default=0
        )
        opens = data["open"].to_numpy(dtype=float)
        closes = data["close"].to_numpy(dtype=float)
        signals = np.zeros(len(data), dtype=int)
        direction = 0
        entry_price = np.nan
        entry_bar = -1
        pending_entry = False
        for i in range(len(data)):
            if pending_entry:
                if not np.isfinite(opens[i]) or opens[i] <= 0:
                    raise ValueError("Entry open must be finite and positive")
                entry_price = opens[i]
                entry_bar = i
                pending_entry = False
            if direction:
                gain = direction * (closes[i] / entry_price - 1.0)
                if gain >= params.profit_target or i - entry_bar + 1 >= params.max_hold_bars:
                    direction = 0
                    entry_price = np.nan
                    entry_bar = -1
                    continue
            elif entries[i] and np.isfinite(closes[i]) and closes[i] > 0:
                direction = int(entries[i])
                pending_entry = True
            signals[i] = direction
        df = pd.DataFrame({"signal": signals, "size": 0.95}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


if __name__ == "__main__":
    index = pd.date_range("2020-01-01", periods=13)
    data = pd.DataFrame({
        "open": 100.0, "high": 106.0, "low": 94.0,
        "close": 100.0, "volume": 1000.0,
    }, index=index)
    data.loc[index[2], "close"] = 105.0
    data.loc[index[6], "close"] = 95.0
    features = pd.DataFrame({"z": 0.0, "pressure": 0.0, "efficiency": 0.5}, index=index)
    features.loc[index[[0, 8]], ["z", "pressure"]] = [2.0, 0.5]
    features.loc[index[4], ["z", "pressure"]] = [-2.0, -0.5]
    strategy = GeneratedStrategy()
    params = RegimeAuctionParams(max_hold_bars=3)
    ctx = StrategyContext(symbol="SPY", timeframe="1d", warmup_bars=41)
    result = strategy.generate_signals(data, features, ctx, params).data
    assert result["signal"].tolist() == [0, 1, 1, 0, 0, -1, -1, 0, 0, 1, 1, 1, 0]
    assert result["size"].gt(0).all()
    flat = strategy.indicators(data.assign(close=100.0), params)
    assert strategy.generate_signals(data, flat, ctx, params).data["signal"].eq(0).all()
