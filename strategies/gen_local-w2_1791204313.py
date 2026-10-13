from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class RangeMigrationParams:
    channel_bars: int = 20
    volume_bars: int = 20
    atr_bars: int = 14
    volume_ratio: float = 1.1
    trail_atr: float = 0.9
    hold_bars: int = 2


class GeneratedStrategy(BaseStrategy[RangeMigrationParams]):
    strategy_id = "gen_local-w2_1791204313"

    @classmethod
    def params_type(cls) -> type[RangeMigrationParams]:
        return RangeMigrationParams

    def warmup_bars(self, params: RangeMigrationParams) -> int:
        return int(max(params.channel_bars, params.volume_bars, params.atr_bars) + 3)

    def indicators(self, data: pd.DataFrame, params: RangeMigrationParams) -> pd.DataFrame:
        if min(params.channel_bars, params.volume_bars, params.atr_bars) < 2:
            raise ValueError("Lookbacks must be at least two bars")
        if params.volume_ratio <= 0 or params.trail_atr <= 0:
            raise ValueError("Volume ratio and ATR multiplier must be positive")
        if params.hold_bars not in (1, 2):
            raise ValueError("Holding horizon must be one or two sessions")
        out = pd.DataFrame(index=data.index)
        close = data["close"]
        out["channel_high"] = data["high"].rolling(params.channel_bars).max().shift(1)
        out["channel_low"] = data["low"].rolling(params.channel_bars).min().shift(1)
        width = (out["channel_high"] - out["channel_low"]).replace(0, np.nan)
        out["position"] = (close - out["channel_low"]) / width
        out["position_change"] = out["position"].diff()
        out["close_change"] = close.diff()
        out["volume_baseline"] = data["volume"].rolling(params.volume_bars).mean().shift(1)
        out["relative_volume"] = data["volume"] / out["volume_baseline"].where(out["volume_baseline"] > 0)
        previous_close = close.shift(1)
        out["true_range"] = pd.concat([
            data["high"] - data["low"],
            (data["high"] - previous_close).abs(),
            (data["low"] - previous_close).abs(),
        ], axis=1).max(axis=1)
        out["atr"] = out["true_range"].rolling(params.atr_bars).mean()
        return out.replace([np.inf, -np.inf], np.nan)

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame,
                         ctx: StrategyContext, params: RangeMigrationParams) -> SignalFrame:
        ready = indicators[["position", "position_change", "close_change", "relative_volume", "atr"]].notna().all(axis=1)
        ready &= indicators["atr"].gt(0)
        participation = indicators["relative_volume"].ge(params.volume_ratio)
        long_step = (ready & participation
                     & indicators["position"].between(0.15, 0.50)
                     & indicators["position_change"].gt(0)
                     & indicators["close_change"].gt(0))
        short_step = (ready & participation
                      & indicators["position"].between(0.50, 0.85)
                      & indicators["position_change"].lt(0)
                      & indicators["close_change"].lt(0))
        long_entry = long_step & long_step.shift(1, fill_value=False)
        short_entry = short_step & short_step.shift(1, fill_value=False)
        entries = np.select([long_entry, short_entry], [1, -1], default=0)
        closes = data["close"].to_numpy(dtype=float)
        atrs = indicators["atr"].to_numpy(dtype=float)
        raw = np.zeros(len(data), dtype=int)
        position = 0
        age = 0
        high_water = np.nan
        low_water = np.nan
        stop = np.nan
        for i in range(len(data)):
            price = closes[i]
            atr = atrs[i]
            if not np.isfinite(price) or not np.isfinite(atr) or atr <= 0:
                position = 0
                age = 0
                continue
            if position:
                age += 1
                if position == 1:
                    high_water = max(high_water, price)
                    stop = max(stop, high_water - params.trail_atr * atr)
                    stopped = price <= stop
                else:
                    low_water = min(low_water, price)
                    stop = min(stop, low_water + params.trail_atr * atr)
                    stopped = price >= stop
                if stopped or age >= params.hold_bars:
                    position = 0
                    age = 0
            elif entries[i]:
                position = int(entries[i])
                age = 0
                high_water = price
                low_water = price
                stop = price - position * params.trail_atr * atr
            raw[i] = position
        df = pd.DataFrame({"signal": raw, "size": 1.0}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


if __name__ == "__main__":
    strategy = GeneratedStrategy()
    params = RangeMigrationParams(channel_bars=3, volume_bars=3, atr_bars=3)
    for direction in (1, -1):
        close = 100.0 + direction * np.array([0, 0, 0, 0, -6, -5, -4, -3, -2, -1, 0, 1])
        data = pd.DataFrame({
            "open": close, "high": 110.0, "low": 90.0, "close": close,
            "volume": [100, 100, 100, 100, 100, 200, 300, 400, 500, 600, 700, 800],
        }, index=pd.date_range("2020-01-01", periods=12))
        result = strategy.generate_signals(data, strategy.indicators(data, params), None, params).data
        assert result["signal"].iloc[:7].eq(0).all()
        assert result["signal"].iloc[7:10].tolist() == [direction, direction, 0]
        stopped_data = data.copy()
        stopped_data.iloc[7, stopped_data.columns.get_loc("close")] = 96.0 - direction
        tight = RangeMigrationParams(channel_bars=3, volume_bars=3, atr_bars=3, trail_atr=0.04)
        stopped = strategy.generate_signals(stopped_data, strategy.indicators(stopped_data, tight), None, tight).data
        assert stopped["signal"].iloc[7:9].tolist() == [direction, 0]
