from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class VolatilityMismatchParams:
    fast_window: int = 3
    slow_window: int = 20
    dispersion_window: int = 20
    range_expansion: float = 1.2
    dispersion_multiple: float = 1.3
    close_threshold: float = 0.75
    trail_atr: float = 0.8
    hold_bars: int = 2


class GeneratedStrategy(BaseStrategy[VolatilityMismatchParams]):
    strategy_id = "gen_local_1791236917"

    @classmethod
    def params_type(cls) -> type[VolatilityMismatchParams]:
        return VolatilityMismatchParams

    def warmup_bars(self, params: VolatilityMismatchParams) -> int:
        return int(max(200, params.fast_window + 1, params.slow_window + 1, params.dispersion_window + 1))

    def indicators(self, data: pd.DataFrame, params: VolatilityMismatchParams) -> pd.DataFrame:
        if min(params.fast_window, params.slow_window, params.dispersion_window, params.hold_bars) < 1:
            raise ValueError("Windows and holding period must be positive")
        if min(params.range_expansion, params.dispersion_multiple, params.trail_atr) <= 0:
            raise ValueError("Volatility multipliers must be positive")
        if not 0 <= params.close_threshold <= 1:
            raise ValueError("close_threshold must be between zero and one")
        out = pd.DataFrame(index=data.index)
        close = data["close"]
        previous_close = close.shift(1)
        out["true_range"] = pd.concat([
            data["high"] - data["low"],
            (data["high"] - previous_close).abs(),
            (data["low"] - previous_close).abs(),
        ], axis=1).max(axis=1)
        out["atr_fast"] = out["true_range"].rolling(params.fast_window, min_periods=params.fast_window).mean()
        out["atr_slow"] = out["true_range"].rolling(params.slow_window, min_periods=params.slow_window).mean()
        out["returns"] = close.pct_change(fill_method=None).replace([np.inf, -np.inf], np.nan)
        out["return_std"] = out["returns"].rolling(params.dispersion_window, min_periods=params.dispersion_window).std(ddof=0)
        out["ma200"] = close.rolling(200, min_periods=200).mean()
        out["range_ratio"] = out["atr_fast"] / out["atr_slow"].replace(0.0, np.nan)
        out["relative_atr"] = out["atr_fast"] / close.replace(0.0, np.nan)
        out["decoupling"] = out["relative_atr"] / out["return_std"].replace(0.0, np.nan)
        out["close_location"] = (close - data["low"]) / (data["high"] - data["low"]).replace(0.0, np.nan)
        out["regime"] = (close > out["ma200"]).fillna(False)
        out["setup"] = (
            out["regime"]
            & (out["range_ratio"] >= params.range_expansion)
            & (out["decoupling"] >= params.dispersion_multiple)
            & (out["returns"] > 0.0)
            & (close > data["open"])
            & (out["close_location"] >= params.close_threshold)
        ).fillna(False)
        out["entry"] = out["setup"] & ~out["setup"].shift(1, fill_value=False)
        return out

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: VolatilityMismatchParams) -> SignalFrame:
        close = data["close"].to_numpy(dtype=float)
        atr = indicators["atr_slow"].to_numpy(dtype=float)
        entry = indicators["entry"].fillna(False).to_numpy(dtype=bool)
        regime = indicators["regime"].fillna(False).to_numpy(dtype=bool)
        signals = np.zeros(len(data), dtype=int)
        active = False
        age = 0
        high_water = -np.inf
        stop = -np.inf
        for i in range(len(data)):
            valid = np.isfinite(close[i]) and np.isfinite(atr[i]) and close[i] > 0.0 and atr[i] > 0.0
            if active:
                age += 1
                if valid:
                    high_water = max(high_water, close[i])
                    stop = max(stop, high_water - params.trail_atr * atr[i])
                if not valid or not regime[i] or close[i] <= stop or age >= params.hold_bars:
                    active = False
                else:
                    signals[i] = 1
            elif valid and regime[i] and entry[i]:
                active = True
                age = 0
                # The next bar is the fill bar; only its close and later closes enter the high-water mark.
                high_water = -np.inf
                stop = -np.inf
                signals[i] = 1
        df = pd.DataFrame({"signal": signals, "size": 1.0}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check() -> None:
    index = pd.date_range("2020-01-01", periods=6)
    close = np.array([100.0, 102.0, 103.0, 101.0, 105.0, 106.0])
    data = pd.DataFrame({"open": close, "high": close + 1.0, "low": close - 1.0, "close": close, "volume": 1000.0}, index=index)
    indicators = pd.DataFrame({"atr_slow": [1.0, 1.0, 5.0, 5.0, 1.0, 1.0], "entry": [True, False, False, False, False, False], "regime": True}, index=index)
    strategy = GeneratedStrategy()
    params = VolatilityMismatchParams(hold_bars=5, trail_atr=1.0)
    result = strategy.generate_signals(data, indicators, None, params).data
    # ATR widening must not lower the existing stop of 101.
    assert result["signal"].tolist() == [0, 1, 1, 1, 0, 0]
    params = VolatilityMismatchParams(hold_bars=2, trail_atr=10.0)
    result = strategy.generate_signals(data, indicators, None, params).data
    assert result["signal"].tolist() == [0, 1, 1, 0, 0, 0]
    assert (result["size"] > 0.0).all()
    assert strategy.warmup_bars(params) >= 200


if __name__ == "__main__":
    _self_check()
