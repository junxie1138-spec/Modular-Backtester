from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class RecoveryParams:
    extreme_window: int = 40
    recovery_streak: int = 3
    displacement: float = 0.05
    pressure_window: int = 5
    pressure_threshold: float = 0.20
    atr_window: int = 14
    breakeven_gain: float = 0.025
    trail_atr: float = 2.5
    max_hold: int = 18


class GeneratedStrategy(BaseStrategy[RecoveryParams]):
    strategy_id = "gen_local_1791285044"

    @classmethod
    def params_type(cls):
        return RecoveryParams

    def warmup_bars(self, params: RecoveryParams) -> int:
        return int(max(params.extreme_window + params.recovery_streak,
                       params.pressure_window, params.atr_window + 1))

    def indicators(self, data: pd.DataFrame, params: RecoveryParams) -> pd.DataFrame:
        if min(params.extreme_window, params.recovery_streak,
               params.pressure_window, params.atr_window, params.max_hold) < 1:
            raise ValueError("Window lengths and holding limit must be positive")
        if not (0 < params.displacement < 1 and
                0 <= params.pressure_threshold <= 1 and
                params.breakeven_gain > 0 and params.trail_atr > 0):
            raise ValueError("Invalid displacement, pressure, or stop parameters")
        close = data["close"]
        change = close.diff()
        up = change.gt(0)
        down = change.lt(0)
        up_streak = up.astype(int).groupby((~up).cumsum()).cumsum()
        down_streak = down.astype(int).groupby((~down).cumsum()).cumsum()
        peak = close.rolling(params.extreme_window).max()
        trough = close.rolling(params.extreme_window).min()
        origin = close.shift(params.recovery_streak)
        origin_drawdown = origin / peak.shift(params.recovery_streak).replace(0, np.nan) - 1
        origin_runup = origin / trough.shift(params.recovery_streak).replace(0, np.nan) - 1
        bar_range = (data["high"] - data["low"]).replace(0, np.nan)
        location = ((2 * close - data["high"] - data["low"]) / bar_range).clip(-1, 1).fillna(0)
        volume = data["volume"].clip(lower=0).fillna(0)
        weighted_pressure = (location * volume).rolling(params.pressure_window).sum()
        volume_sum = volume.rolling(params.pressure_window).sum()
        pressure = (weighted_pressure / volume_sum.replace(0, np.nan)).fillna(0)
        previous_close = close.shift(1)
        true_range = pd.concat([
            data["high"] - data["low"],
            (data["high"] - previous_close).abs(),
            (data["low"] - previous_close).abs(),
        ], axis=1).max(axis=1)
        atr = true_range.rolling(params.atr_window).mean()
        return pd.DataFrame({
            "change": change,
            "up_streak": up_streak,
            "down_streak": down_streak,
            "peak": peak,
            "trough": trough,
            "origin": origin,
            "origin_drawdown": origin_drawdown,
            "origin_runup": origin_runup,
            "location": location,
            "weighted_pressure": weighted_pressure,
            "volume_sum": volume_sum,
            "pressure": pressure,
            "true_range": true_range,
            "atr": atr,
        }, index=data.index)

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame,
                         ctx: StrategyContext, params: RecoveryParams) -> SignalFrame:
        long_recovery = (
            indicators["up_streak"].eq(params.recovery_streak)
            & indicators["origin_drawdown"].le(-params.displacement)
        )
        short_recovery = (
            indicators["down_streak"].eq(params.recovery_streak)
            & indicators["origin_runup"].ge(params.displacement)
        )
        long_entry = long_recovery & indicators["pressure"].ge(params.pressure_threshold)
        short_entry = short_recovery & indicators["pressure"].le(-params.pressure_threshold)
        candidates = np.select([long_entry, short_entry], [1, -1], default=0)
        opens = data["open"].to_numpy(dtype=float)
        closes = data["close"].to_numpy(dtype=float)
        atr = indicators["atr"].to_numpy(dtype=float)
        raw = np.zeros(len(data), dtype=int)
        position = 0
        entry_price = 0.0
        entry_bar = 0
        stop = -np.inf
        activated = False
        warmup = self.warmup_bars(params)

        for i in range(len(data)):
            # A decision on the previous close becomes a position at this open.
            if position == 0 and i > 0 and raw[i - 1] != 0:
                if not (np.isfinite(opens[i]) and opens[i] > 0
                        and np.isfinite(atr[i - 1]) and atr[i - 1] > 0):
                    continue
                position = int(raw[i - 1])
                entry_price = opens[i]
                entry_bar = i
                activated = False
                # Favorable-price coordinates: the stop rises for either direction.
                stop = position * entry_price - params.trail_atr * atr[i - 1]

            if position != 0:
                if not (np.isfinite(closes[i]) and closes[i] > 0
                        and np.isfinite(atr[i]) and atr[i] > 0):
                    raw[i] = position
                    continue
                favorable_price = position * closes[i]
                gain = position * (closes[i] / entry_price - 1)
                if gain >= params.breakeven_gain:
                    activated = True
                if activated:
                    stop = max(stop, position * entry_price,
                               favorable_price - params.trail_atr * atr[i])
                # Stops use completed closes; exits execute at the next open.
                if favorable_price <= stop or i - entry_bar + 1 >= params.max_hold:
                    position = 0
                    raw[i] = 0
                else:
                    raw[i] = position
                continue

            if i >= warmup - 1 and np.isfinite(atr[i]) and atr[i] > 0:
                raw[i] = int(candidates[i])

        df = pd.DataFrame({"signal": raw, "size": 0.95}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check():
    index = pd.date_range("2020-01-01", periods=6)
    data = pd.DataFrame({
        "open": [100., 100., 101., 104., 102., 101.],
        "high": [101., 102., 105., 105., 103., 102.],
        "low": [99., 99., 100., 101., 100., 100.],
        "close": [100., 101., 104., 102., 101., 101.],
        "volume": [100.] * 6,
    }, index=index)
    params = RecoveryParams(extreme_window=1, recovery_streak=1,
                            pressure_window=1, atr_window=1,
                            breakeven_gain=0.02, trail_atr=1.0)
    ind = pd.DataFrame({
        "up_streak": [0, 1, 0, 0, 0, 0],
        "down_streak": [0] * 6,
        "origin_drawdown": [-0.1] * 6,
        "origin_runup": [0.] * 6,
        "pressure": [1.] * 6,
        "atr": [2.] * 6,
    }, index=index)
    strategy = GeneratedStrategy()
    result = strategy.generate_signals(data, ind, None, params).data
    assert result["signal"].tolist() == [0, 0, 1, 1, 0, 0]
    mirrored = data.copy()
    mirrored["open"] = 200 - data["open"]
    mirrored["close"] = 200 - data["close"]
    mirrored["high"] = 200 - data["low"]
    mirrored["low"] = 200 - data["high"]
    short_ind = ind.copy()
    short_ind["up_streak"] = 0
    short_ind["down_streak"] = ind["up_streak"]
    short_ind["origin_drawdown"] = 0.0
    short_ind["origin_runup"] = 0.1
    short_ind["pressure"] = -1.0
    result = strategy.generate_signals(mirrored, short_ind, None, params).data
    assert result["signal"].tolist() == [0, 0, -1, -1, 0, 0]


if __name__ == "__main__":
    _self_check()
