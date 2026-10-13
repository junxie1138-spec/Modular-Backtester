from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class PlateauParams:
    depth_window: int = 63
    atr_window: int = 14
    depth_atr: float = 3.5
    plateau_atr: float = 0.45
    rejection_location: float = 0.65
    trail_atr: float = 3.0
    hold_bars: int = 18


class GeneratedStrategy(BaseStrategy[PlateauParams]):
    strategy_id = "gen_local_1791268703"

    @classmethod
    def params_type(cls):
        return PlateauParams

    def warmup_bars(self, params: PlateauParams) -> int:
        return max(params.depth_window + 1, params.atr_window + 1)

    def indicators(self, data: pd.DataFrame, params: PlateauParams) -> pd.DataFrame:
        if (params.depth_window < 2 or params.atr_window < 2
                or params.hold_bars < 1 or params.depth_atr <= 0
                or params.plateau_atr <= 0 or params.trail_atr <= 0
                or not 0.5 < params.rejection_location < 1):
            raise ValueError("Invalid strategy parameters")
        close = data["close"]
        previous = close.shift(1)
        tr = pd.concat([data["high"] - data["low"],
                        (data["high"] - previous).abs(),
                        (data["low"] - previous).abs()], axis=1).max(axis=1)
        out = pd.DataFrame(index=data.index)
        out["atr"] = tr.rolling(params.atr_window, min_periods=params.atr_window).mean().replace(0, np.nan)
        out["peak"] = close.rolling(params.depth_window, min_periods=params.depth_window).max()
        out["trough"] = close.rolling(params.depth_window, min_periods=params.depth_window).min()
        out["drawdown"] = (out["peak"] - close) / out["atr"]
        out["drawup"] = (close - out["trough"]) / out["atr"]
        out["movement"] = close.diff().abs() / out["atr"]
        span = (data["high"] - data["low"]).replace(0, np.nan)
        out["location"] = (close - data["low"]) / span
        valid = np.isfinite(out[["atr", "drawdown", "drawup", "movement", "location"]]).all(axis=1)
        plateau = out["movement"] <= params.plateau_atr
        out["long_setup"] = (valid & plateau & (out["drawdown"] >= params.depth_atr)
                             & (out["drawdown"] > out["drawup"])
                             & (out["location"] >= params.rejection_location))
        out["short_setup"] = (valid & plateau & (out["drawup"] >= params.depth_atr)
                              & (out["drawup"] > out["drawdown"])
                              & (out["location"] <= 1 - params.rejection_location))
        out["long_entry"] = out["long_setup"] & out["long_setup"].shift(1, fill_value=False)
        out["short_entry"] = out["short_setup"] & out["short_setup"].shift(1, fill_value=False)
        return out

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame,
                         ctx: StrategyContext, params: PlateauParams) -> SignalFrame:
        close = data["close"].to_numpy(dtype=float)
        atr = indicators["atr"].to_numpy(dtype=float)
        longs = indicators["long_entry"].to_numpy(dtype=bool)
        shorts = indicators["short_entry"].to_numpy(dtype=bool)
        signals = np.zeros(len(data), dtype=int)
        position = 0
        entry_bar = -1
        high_water = -np.inf
        low_water = np.inf
        trailing_floor = -np.inf
        short_ceiling = np.inf
        for i in range(len(data)):
            if not (np.isfinite(close[i]) and np.isfinite(atr[i]) and atr[i] > 0):
                position = 0
                continue
            if position:
                # Entry fills on this bar or earlier; only in-trade closes update the trail.
                high_water = max(high_water, close[i])
                low_water = min(low_water, close[i])
                trailing_floor = max(trailing_floor, high_water - params.trail_atr * atr[i])
                short_ceiling = min(short_ceiling, low_water + params.trail_atr * atr[i])
                # The required close-high drawdown trail applies to both directions.
                # Shorts also cover when their favorable close-low trail is breached.
                if (close[i] <= trailing_floor
                        or (position == -1 and close[i] >= short_ceiling)
                        or i - entry_bar + 1 >= params.hold_bars):
                    position = 0
            elif longs[i] or shorts[i]:
                position = 1 if longs[i] else -1
                entry_bar = i + 1
                high_water = -np.inf
                low_water = np.inf
                trailing_floor = -np.inf
                short_ceiling = np.inf
            signals[i] = position
        df = pd.DataFrame({"signal": signals, "size": 0.95}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check():
    strategy = GeneratedStrategy()
    params = PlateauParams(depth_window=6, atr_window=2, depth_atr=1.5,
                           plateau_atr=0.5, trail_atr=2.0)
    index = pd.date_range("2020-01-01", periods=9)
    close = np.array([100, 100, 100, 100, 90, 90, 90, 90, 90], dtype=float)
    data = pd.DataFrame({"open": close, "high": close + 0.2,
                         "low": close - 0.8, "close": close, "volume": 1000.0}, index=index)
    derived = strategy.indicators(data, params)
    assert not derived["long_entry"].iloc[5]
    assert derived["long_entry"].iloc[6]
    result = strategy.generate_signals(data, derived, None, params).data
    assert result["signal"].iloc[6] == 0 and result["signal"].iloc[7] == 1
    probe = data.iloc[:6].copy()
    probe["close"] = [100.0, 100.0, 104.0, 101.0, 101.0, 101.0]
    manual = pd.DataFrame({"atr": [1.0, 1.0, 1.0, 5.0, 1.0, 1.0],
                           "long_entry": [True, False, False, False, False, False],
                           "short_entry": False}, index=probe.index)
    result = strategy.generate_signals(probe, manual, None, params).data
    assert result["signal"].tolist() == [0, 1, 1, 1, 0, 0]
    manual["long_entry"] = False
    manual["short_entry"] = [True, False, False, False, False, False]
    probe["close"] = [100.0, 100.0, 99.0, 102.0, 102.0, 102.0]
    result = strategy.generate_signals(probe, manual, None, params).data
    assert result["signal"].tolist() == [0, -1, -1, -1, 0, 0]


if __name__ == "__main__":
    _self_check()
