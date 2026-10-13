from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class GapPressureParams:
    atr_window: int = 14
    pressure_window: int = 5
    trend_window: int = 10
    gap_threshold: float = 0.4
    retention_threshold: float = 0.75
    efficiency_threshold: float = 0.35
    trail_atr: float = 0.8
    hold_bars: int = 2


class GeneratedStrategy(BaseStrategy[GapPressureParams]):
    strategy_id = "gen_local_1791283357"

    @classmethod
    def params_type(cls) -> type[GapPressureParams]:
        return GapPressureParams

    def warmup_bars(self, params: GapPressureParams) -> int:
        return int(max(200, params.atr_window + 2,
                       params.pressure_window + 2, params.trend_window + 1))

    def indicators(self, data: pd.DataFrame, params: GapPressureParams) -> pd.DataFrame:
        if min(params.atr_window, params.pressure_window, params.trend_window) < 1:
            raise ValueError("Indicator windows must be positive")
        thresholds = np.array([params.gap_threshold, params.retention_threshold,
                               params.efficiency_threshold, params.trail_atr], dtype=float)
        if not np.isfinite(thresholds).all() or (thresholds <= 0).any():
            raise ValueError("Thresholds must be finite and positive")
        if params.efficiency_threshold > 1 or params.hold_bars not in (1, 2):
            raise ValueError("Efficiency must not exceed one; hold_bars must be one or two")

        close = data["close"]
        previous_close = close.shift(1)
        out = pd.DataFrame(index=data.index)
        out["gap"] = data["open"] - previous_close
        out["true_range"] = pd.concat([
            data["high"] - data["low"],
            (data["high"] - previous_close).abs(),
            (data["low"] - previous_close).abs(),
        ], axis=1).max(axis=1, skipna=False)
        out["atr"] = out["true_range"].rolling(params.atr_window).mean()
        out["previous_atr"] = out["atr"].shift(1)
        out["gap_atr"] = out["gap"] / out["previous_atr"].where(out["previous_atr"] > 0)
        out["prior_gap_pressure"] = out["gap_atr"].rolling(params.pressure_window).sum().shift(1)
        out["retention"] = (close - previous_close) / out["gap"].where(out["gap"] > 0)
        out["close_change"] = close.diff()
        out["path_length"] = out["close_change"].abs().rolling(params.trend_window).sum()
        out["trend_displacement"] = close.diff(params.trend_window)
        out["efficiency"] = out["trend_displacement"] / out["path_length"].where(out["path_length"] > 0)
        out["ma200"] = close.rolling(200).mean()
        out = out.replace([np.inf, -np.inf], np.nan)
        ready = out[["atr", "gap_atr", "prior_gap_pressure", "retention",
                     "efficiency", "ma200"]].notna().all(axis=1)
        out["entry"] = (
            ready
            & (out["atr"] > 0)
            & (close > out["ma200"])
            & (out["efficiency"] >= params.efficiency_threshold)
            & (out["prior_gap_pressure"] < 0)
            & (out["gap_atr"] >= params.gap_threshold)
            & (out["retention"] >= params.retention_threshold)
        ).fillna(False)
        return out

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame,
                         ctx: StrategyContext, params: GapPressureParams) -> SignalFrame:
        if params.hold_bars not in (1, 2) or not np.isfinite(params.trail_atr) or params.trail_atr <= 0:
            raise ValueError("Invalid holding period or ATR trail")
        closes = data["close"].to_numpy(dtype=float)
        atrs = indicators["atr"].to_numpy(dtype=float)
        entries = indicators["entry"].fillna(False).to_numpy(dtype=bool)
        signals = np.zeros(len(data), dtype=int)
        active = False
        age = 0
        high_water = -np.inf
        stop = -np.inf

        for i in range(len(data)):
            close, atr = closes[i], atrs[i]
            valid = np.isfinite(close) and close > 0 and np.isfinite(atr) and atr > 0
            if active:
                # Entry filled at this bar's open when age is zero.
                age += 1
                if valid:
                    high_water = max(high_water, close)
                    stop = max(stop, high_water - params.trail_atr * atr)
                if not valid or close <= stop or age >= params.hold_bars:
                    active = False
                    age = 0
                    high_water = stop = -np.inf
            elif entries[i] and valid:
                # The decision close precedes entry and is excluded from the high-water mark.
                active = True
                age = 0
                high_water = stop = -np.inf
            signals[i] = int(active)

        df = pd.DataFrame({"signal": signals, "size": 1.0}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


if __name__ == "__main__":
    index = pd.date_range("2020-01-01", periods=7, freq="D")
    sample = pd.DataFrame({"close": [100., 102., 101., 100., 101., 102., 103.]}, index=index)
    derived = pd.DataFrame({"atr": 1., "entry": [True, True, True, True, False, False, False]}, index=index)
    strategy = GeneratedStrategy()
    result = strategy.generate_signals(sample, derived, None, GapPressureParams())
    assert result.data["signal"].tolist() == [0, 1, 1, 0, 1, 1, 0]
    assert (result.data["size"] > 0).all()
