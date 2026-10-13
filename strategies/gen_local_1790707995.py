from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class GeneratedParams:
    channel_bars: int = 40
    migration_bars: int = 5
    volume_threshold: float = 0.30
    atr_bars: int = 14
    breakeven_pct: float = 0.03
    trail_atr: float = 2.5
    max_hold: int = 20
    size: float = 1.0


class GeneratedStrategy(BaseStrategy[GeneratedParams]):
    strategy_id = "gen_local_1790707995"

    @classmethod
    def params_type(cls):
        return GeneratedParams

    def warmup_bars(self, params: GeneratedParams) -> int:
        return max(params.channel_bars + params.migration_bars, params.atr_bars + 1)

    def indicators(self, data: pd.DataFrame, params: GeneratedParams) -> pd.DataFrame:
        close = data["close"]
        volume = data["volume"]
        floor = data["low"].rolling(params.channel_bars).min()
        ceiling = data["high"].rolling(params.channel_bars).max()
        position = ((close - floor) / (ceiling - floor).replace(0, np.nan)).fillna(0.5)
        signed_volume = np.sign(close.diff().fillna(0)) * volume
        pressure = (signed_volume.rolling(params.migration_bars).sum() / volume.rolling(params.migration_bars).sum().replace(0, np.nan)).fillna(0.0)
        previous_close = close.shift(1)
        true_range = pd.concat([data["high"] - data["low"], (data["high"] - previous_close).abs(), (data["low"] - previous_close).abs()], axis=1).max(axis=1)
        atr = true_range.rolling(params.atr_bars).mean().fillna(0.0)
        return pd.DataFrame({"position": position, "pressure": pressure, "atr": atr}, index=data.index)

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: GeneratedParams) -> SignalFrame:
        current = indicators["position"].to_numpy()
        previous = indicators["position"].shift(params.migration_bars).fillna(0.5).to_numpy()
        pressure = indicators["pressure"].to_numpy()
        atr = indicators["atr"].to_numpy()
        opens = data["open"].to_numpy()
        closes = data["close"].to_numpy()
        raw = np.zeros(len(data), dtype=int)
        side = 0
        age = 0
        entry = stop = peak = 0.0
        armed = False
        for i in range(len(data)):
            exited = False
            if side:
                if age == 0:
                    entry = float(opens[i])
                    peak = entry
                    stop = entry - side * params.trail_atr * atr[i]
                age += 1
                price = float(closes[i])
                if side == 1:
                    peak = max(peak, price)
                    armed = armed or price >= entry * (1 + params.breakeven_pct)
                    if armed:
                        stop = max(stop, entry, peak - params.trail_atr * atr[i])
                    exited = price <= stop
                else:
                    peak = min(peak, price)
                    armed = armed or price <= entry * (1 - params.breakeven_pct)
                    if armed:
                        stop = min(stop, entry, peak + params.trail_atr * atr[i])
                    exited = price >= stop
                if exited or age >= params.max_hold:
                    side = 0
                    age = 0
                    exited = True
            if not side and not exited and i >= self.warmup_bars(params):
                if previous[i] <= 0.35 and current[i] >= 0.70 and pressure[i] >= params.volume_threshold:
                    side = 1
                elif previous[i] >= 0.65 and current[i] <= 0.30 and pressure[i] <= -params.volume_threshold:
                    side = -1
                if side:
                    armed = False
                    age = 0
            raw[i] = side
        df = pd.DataFrame({"signal": raw, "size": float(params.size)}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


if __name__ == "__main__":
    p = GeneratedParams(channel_bars=3, migration_bars=2, volume_threshold=0.2, atr_bars=2, breakeven_pct=0.05, trail_atr=1.0)
    prices = pd.Series([10, 10, 10, 9, 8, 9, 10, 11, 12, 13, 14, 15, 14, 13, 10, 10], dtype=float)
    data = pd.DataFrame({"open": prices, "high": prices + 0.1, "low": prices - 0.1, "close": prices, "volume": 100.0})
    strategy = GeneratedStrategy()
    signals = strategy.generate_signals(data, strategy.indicators(data, p), StrategyContext("SPY", "1d", strategy.warmup_bars(p)), p).data["signal"]
    assert signals.iloc[6] == 0 and signals.iloc[7] == 1
    assert ((signals.shift(1) == 1) & (signals == 0)).any()
