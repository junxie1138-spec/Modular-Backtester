from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class RangeParams:
    contraction_ratio: float = 0.60
    trail_atr: float = 1.80


class GeneratedStrategy(BaseStrategy[RangeParams]):
    strategy_id = "gen_local_1791266073"

    @classmethod
    def params_type(cls) -> type[RangeParams]:
        return RangeParams

    def warmup_bars(self, params: RangeParams) -> int:
        return 24

    def indicators(self, data: pd.DataFrame, params: RangeParams) -> pd.DataFrame:
        if not 0.0 < params.contraction_ratio < 1.0 or not np.isfinite(params.trail_atr) or params.trail_atr <= 0.0:
            raise ValueError("Require 0 < contraction_ratio < 1 and finite trail_atr > 0")
        out = pd.DataFrame(index=data.index)
        previous = data["close"].shift(1).replace(0.0, np.nan)
        out["range"] = (data["high"] - data["low"]).clip(lower=0.0)
        out["relative_range"] = out["range"] / previous
        out["baseline"] = out["relative_range"].rolling(20, min_periods=20).median().shift(3)
        out["recent_range"] = out["relative_range"].rolling(3, min_periods=3).mean()
        out["range_ratio"] = out["recent_range"] / out["baseline"].replace(0.0, np.nan)
        out["close_location"] = (data["close"] - data["low"]) / out["range"].replace(0.0, np.nan)
        out["acceptance"] = out["close_location"].rolling(3, min_periods=3).min()
        out["three_day_gain"] = data["close"] / data["close"].shift(3).replace(0.0, np.nan) - 1.0
        out["true_range"] = pd.concat([
            out["range"],
            (data["high"] - previous).abs(),
            (data["low"] - previous).abs(),
        ], axis=1).max(axis=1)
        out["atr"] = out["true_range"].rolling(14, min_periods=14).mean()
        out["entry"] = (
            (out["range_ratio"] < params.contraction_ratio)
            & (out["range_ratio"].shift(1) >= params.contraction_ratio)
            & (out["acceptance"] >= 0.55)
            & (out["three_day_gain"] > 0.0)
            & (out["atr"] > 0.0)
        ).fillna(False)
        return out.replace([np.inf, -np.inf], np.nan)

    def generate_signals(
        self, data: pd.DataFrame, indicators: pd.DataFrame,
        ctx: StrategyContext, params: RangeParams,
    ) -> SignalFrame:
        df = pd.DataFrame(index=data.index)
        signal = np.zeros(len(data), dtype=int)
        prices = data[["open", "high", "low", "close"]].to_numpy(dtype=float)
        atr = indicators["atr"].to_numpy(dtype=float)
        entries = indicators["entry"].fillna(False).to_numpy(dtype=bool)
        active = pending = armed = False
        entry = peak = stop = 0.0
        age = next_entry = 0
        for i, (opening, high, low, close) in enumerate(prices):
            valid = np.isfinite(prices[i]).all() and min(opening, high, low, close) > 0.0
            if pending:
                pending = False
                if valid and i > 0 and np.isfinite(atr[i - 1]) and atr[i - 1] > 0.0:
                    # Reference entry is this bar's open, after yesterday's decision.
                    active, armed, age = True, False, 0
                    entry = peak = opening
                    stop = entry - params.trail_atr * atr[i - 1]
            if active:
                age += 1
                exit_now = not valid or opening <= stop or low <= stop
                if not exit_now:
                    peak = max(peak, high)
                    armed = armed or high >= entry * 1.008
                    if armed:
                        stop = max(stop, entry)
                        if np.isfinite(atr[i]) and atr[i] > 0.0:
                            stop = max(stop, peak - params.trail_atr * atr[i])
                    # New stops apply to the close and future bars, never earlier lows.
                    exit_now = close <= stop or age >= 4
                if exit_now:
                    active = False
                    next_entry = i + 5
                else:
                    signal[i] = 1
            elif valid and i >= next_entry and entries[i]:
                pending = True
                signal[i] = 1
        # Stop breaches are close-time exit decisions filled next bar.
        df["signal"] = signal
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = 1.0
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check() -> None:
    index = pd.date_range("2020-01-01", periods=6)
    data = pd.DataFrame({
        "open": [100.0, 100.0, 100.2, 101.6, 101.5, 101.5],
        "high": [100.2, 100.4, 102.0, 103.0, 102.0, 102.0],
        "low": [99.8, 99.7, 100.1, 101.0, 101.0, 101.0],
        "close": [100.0, 100.2, 101.6, 101.8, 101.5, 101.5],
        "volume": 1000.0,
    }, index=index)
    indicators = pd.DataFrame({
        "atr": [1.0, 1.0, 0.5, 4.0, 4.0, 4.0],
        "entry": [True, False, False, False, False, False],
    }, index=index)
    strategy = GeneratedStrategy()
    result = strategy.generate_signals(data, indicators, None, RangeParams()).data
    assert result["signal"].tolist() == [0, 1, 1, 1, 0, 0]
    assert (result["size"] > 0.0).all()
    # Without a stop breach, the fourth held session schedules the exit.
    data["high"] = 100.4
    data["low"] = 99.9
    data["close"] = 100.1
    data["open"] = 100.0
    indicators["atr"] = 1.0
    result = strategy.generate_signals(data, indicators, None, RangeParams()).data
    assert result["signal"].tolist() == [0, 1, 1, 1, 1, 0]


if __name__ == "__main__":
    _self_check()
