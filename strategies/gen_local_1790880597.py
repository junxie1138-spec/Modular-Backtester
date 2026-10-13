from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class StrategyParams:
    streak_length: int = 3
    flow_window: int = 10
    flow_threshold: float = 0.20
    atr_period: int = 14
    stop_atr: float = 2.5
    hold_bars: int = 18

    def __post_init__(self):
        if min(self.streak_length, self.flow_window, self.atr_period, self.hold_bars) < 1:
            raise ValueError("Lengths must be positive.")
        if not 0.0 < self.flow_threshold <= 1.0 or not np.isfinite(self.stop_atr) or self.stop_atr <= 0:
            raise ValueError("Invalid flow threshold or stop multiplier.")


class GeneratedStrategy(BaseStrategy[StrategyParams]):
    strategy_id = "gen_local_1790880597"

    @classmethod
    def params_type(cls):
        return StrategyParams

    def warmup_bars(self, params):
        return int(max(params.flow_window, params.atr_period, params.streak_length) + 1)

    def indicators(self, data, params):
        close = data["close"]
        volume = data["volume"].clip(lower=0)
        change = close.diff()
        quieter = volume.lt(volume.shift(1))
        up = change.gt(0) & quieter
        down = change.lt(0) & quieter
        up_streak = up.astype(int).groupby((~up).cumsum()).cumsum()
        down_streak = down.astype(int).groupby((~down).cumsum()).cumsum()
        span = (data["high"] - data["low"]).replace(0, np.nan)
        location = ((2 * close - data["high"] - data["low"]) / span).clip(-1, 1).fillna(0)
        flow_volume = volume.rolling(params.flow_window, min_periods=params.flow_window).sum()
        flow = (location * volume).rolling(params.flow_window, min_periods=params.flow_window).sum() / flow_volume.replace(0, np.nan)
        previous = close.shift(1)
        tr = pd.concat([
            data["high"] - data["low"],
            (data["high"] - previous).abs(),
            (data["low"] - previous).abs(),
        ], axis=1).max(axis=1)
        atr = tr.rolling(params.atr_period, min_periods=params.atr_period).mean()
        return pd.DataFrame({
            "close_change": change,
            "quieter_volume": quieter,
            "up_streak": up_streak,
            "down_streak": down_streak,
            "close_location": location,
            "flow_volume": flow_volume,
            "flow": flow,
            "true_range": tr,
            "atr": atr,
        }, index=data.index)

    def generate_signals(self, data, indicators, ctx, params):
        long_entry = (indicators["up_streak"] >= params.streak_length) & (indicators["flow"] >= params.flow_threshold)
        short_entry = (indicators["down_streak"] >= params.streak_length) & (indicators["flow"] <= -params.flow_threshold)
        candidate = np.select([long_entry, short_entry], [1, -1], default=0)
        opens = data["open"].to_numpy(dtype=float)
        closes = data["close"].to_numpy(dtype=float)
        atr = indicators["atr"].to_numpy(dtype=float)
        signal = np.zeros(len(data), dtype=int)
        direction = 0
        pending = False
        stop = np.nan
        entry_bar = -1
        frozen_atr = np.nan
        for i in range(len(data)):
            if direction:
                if pending:
                    # The previous close ordered entry at this bar's open.
                    if not np.isfinite(opens[i]) or opens[i] <= 0:
                        direction = 0
                        pending = False
                        continue
                    stop = opens[i] - direction * params.stop_atr * frozen_atr
                    entry_bar = i
                    pending = False
                stopped = np.isfinite(closes[i]) and direction * (closes[i] - stop) < 0
                expired = i - entry_bar + 1 >= params.hold_bars
                if stopped or expired or not np.isfinite(closes[i]):
                    direction = 0
                signal[i] = direction
                continue
            if i < self.warmup_bars(params) - 1:
                continue
            if candidate[i] and np.isfinite(closes[i]) and closes[i] > 0 and np.isfinite(atr[i]) and atr[i] > 0:
                direction = int(candidate[i])
                frozen_atr = atr[i]
                pending = True
                signal[i] = direction
        df = pd.DataFrame({"signal": signal, "size": 0.95}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check():
    index = pd.date_range("2020-01-01", periods=7)
    data = pd.DataFrame({
        "open": [100, 100, 102, 100, 100, 98, 100],
        "close": [100, 100, 99, 100, 100, 101, 100],
        "high": [103.0] * 7,
        "low": [97.0] * 7,
        "volume": [1000.0] * 7,
    }, index=index)
    indicators = pd.DataFrame({
        "up_streak": [0, 3, 0, 0, 0, 0, 0],
        "down_streak": [0, 0, 0, 0, 3, 0, 0],
        "flow": [0, 0.5, 0, 0, -0.5, 0, 0],
        "atr": [1.0] * 7,
    }, index=index)
    params = StrategyParams(streak_length=1, flow_window=1, atr_period=1)
    strategy = GeneratedStrategy()
    ctx = StrategyContext(symbol="SPY", timeframe="1d", warmup_bars=2)
    result = strategy.generate_signals(data, indicators, ctx, params).data
    assert result["signal"].tolist() == [0, 0, 1, 0, 0, -1, 0]
    params.hold_bars = 2
    indicators["atr"] = 100.0
    result = strategy.generate_signals(data, indicators, ctx, params).data
    assert result["signal"].tolist() == [0, 0, 1, 1, 0, -1, -1]
    derived = strategy.indicators(data, params)
    assert derived.index.equals(data.index)
    assert result["size"].gt(0).all()


if __name__ == "__main__":
    _self_check()
