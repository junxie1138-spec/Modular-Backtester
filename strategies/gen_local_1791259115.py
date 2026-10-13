from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class RangeCycleParams:
    regime_window: int = 30
    fast_window: int = 3
    slow_window: int = 15
    pressure_window: int = 5
    atr_window: int = 14
    regime_threshold: float = 0.15
    pressure_threshold: float = 0.15
    trail_atr: float = 2.0
    max_hold_bars: int = 10


class GeneratedStrategy(BaseStrategy[RangeCycleParams]):
    strategy_id = "gen_local_1791259115"

    @classmethod
    def params_type(cls) -> type[RangeCycleParams]:
        return RangeCycleParams

    def warmup_bars(self, params: RangeCycleParams) -> int:
        return int(max(params.regime_window + 2, params.fast_window + 2,
                       params.slow_window + 1, params.pressure_window,
                       params.atr_window + 1))

    def indicators(self, data: pd.DataFrame, params: RangeCycleParams) -> pd.DataFrame:
        windows = (params.regime_window, params.fast_window, params.slow_window,
                   params.pressure_window, params.atr_window, params.max_hold_bars)
        if any(w < 1 for w in windows) or params.regime_window < 3:
            raise ValueError("Windows must be positive and regime_window at least three.")
        thresholds = (params.regime_threshold, params.pressure_threshold, params.trail_atr)
        if not all(np.isfinite(x) for x in thresholds):
            raise ValueError("Thresholds must be finite.")
        if not 0 <= params.regime_threshold < 1 or not 0 <= params.pressure_threshold < 1 or params.trail_atr <= 0:
            raise ValueError("Invalid regime, pressure, or trailing-stop threshold.")
        out = pd.DataFrame(index=data.index)
        out["previous_close"] = data["close"].shift(1)
        out["daily_range"] = (data["high"] - data["low"]).clip(lower=0)
        out["range_fraction"] = out["daily_range"].div(out["previous_close"].where(out["previous_close"] > 0))
        out["lagged_range"] = out["range_fraction"].shift(1)
        out["range_memory"] = out["range_fraction"].rolling(params.regime_window).corr(out["lagged_range"])
        out["fast_range"] = out["range_fraction"].rolling(params.fast_window).mean()
        out["slow_range"] = out["range_fraction"].rolling(params.slow_window).mean()
        out["range_ratio"] = out["fast_range"].div(out["slow_range"].where(out["slow_range"] > 0))
        out["range_change"] = out["fast_range"].diff()
        out["body_pressure"] = (data["close"] - data["open"]).div(out["daily_range"].where(out["daily_range"] > 0)).fillna(0).clip(-1, 1)
        out["buying_pressure"] = out["body_pressure"].rolling(params.pressure_window).mean()
        out["high_gap"] = (data["high"] - out["previous_close"]).abs()
        out["low_gap"] = (data["low"] - out["previous_close"]).abs()
        out["true_range"] = out[["daily_range", "high_gap", "low_gap"]].max(axis=1)
        out["atr"] = out["true_range"].rolling(params.atr_window).mean()
        out = out.replace([np.inf, -np.inf], np.nan)
        out["persistent_regime"] = out["range_memory"].gt(params.regime_threshold)
        out["alternating_regime"] = out["range_memory"].lt(-params.regime_threshold)
        out["range_agrees"] = (
            (out["persistent_regime"] & out["range_ratio"].gt(1) & out["range_change"].gt(0))
            | (out["alternating_regime"] & out["range_ratio"].lt(1) & out["range_change"].lt(0))
        )
        out["pressure_agrees"] = out["buying_pressure"].gt(params.pressure_threshold)
        out["entry"] = out["range_agrees"] & out["pressure_agrees"] & out["atr"].gt(0)
        return out

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame,
                         ctx: StrategyContext, params: RangeCycleParams) -> SignalFrame:
        close = data["close"].to_numpy(dtype=float)
        atr = indicators["atr"].to_numpy(dtype=float)
        entries = indicators["entry"].fillna(False).to_numpy(dtype=bool)
        signals = np.zeros(len(data), dtype=int)
        active = False
        high_water = np.nan
        stop = -np.inf
        age = 0
        for i in range(len(data)):
            valid = np.isfinite(close[i]) and close[i] > 0 and np.isfinite(atr[i]) and atr[i] > 0
            if active:
                age += 1
                if not valid:
                    active = False
                else:
                    high_water = close[i] if np.isnan(high_water) else max(high_water, close[i])
                    stop = max(stop, high_water - params.trail_atr * atr[i])
                    if close[i] <= stop or age >= params.max_hold_bars:
                        active = False
            elif valid and entries[i]:
                active = True
                age = 0
                # Entry fills next bar; initialize the close watermark on that bar.
                high_water = np.nan
                stop = -np.inf
            signals[i] = int(active)
        df = pd.DataFrame({"signal": signals, "size": 1.0}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check() -> None:
    index = pd.date_range("2020-01-01", periods=5)
    data = pd.DataFrame({"close": [100.0, 103.0, 102.0, 100.0, 110.0]}, index=index)
    indicators = pd.DataFrame({"atr": [1.0, 1.0, 5.0, 5.0, 1.0], "entry": True}, index=index)
    result = GeneratedStrategy.generate_signals(None, data, indicators, None, RangeCycleParams())
    # ATR expansion must not lower the stop; the exit is executed one bar later.
    assert result.data["signal"].tolist() == [0, 1, 1, 1, 0]
    assert result.data["size"].gt(0).all()


if __name__ == "__main__":
    _self_check()
