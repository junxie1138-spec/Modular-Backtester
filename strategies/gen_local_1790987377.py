from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class RangePulseParams:
    shock_multiple: float = 1.5
    trail_atr: float = 2.5


class GeneratedStrategy(BaseStrategy[RangePulseParams]):
    strategy_id = "gen_local_1790987377"

    @classmethod
    def params_type(cls):
        return RangePulseParams

    def warmup_bars(self, params: RangePulseParams) -> int:
        return 23

    def indicators(self, data: pd.DataFrame, params: RangePulseParams) -> pd.DataFrame:
        out = pd.DataFrame(index=data.index)
        out["range"] = (data["high"] - data["low"]).clip(lower=0.0)
        out["baseline"] = out["range"].rolling(20, min_periods=20).median().shift(1)
        out["previous_close"] = data["close"].shift(1)
        out["true_range"] = pd.concat(
            [out["range"], (data["high"] - out["previous_close"]).abs(),
             (data["low"] - out["previous_close"]).abs()], axis=1
        ).max(axis=1)
        out["atr"] = out["true_range"].rolling(14, min_periods=14).mean()
        out["close_location"] = (
            (data["close"] - data["low"]) / out["range"].replace(0.0, np.nan)
        )
        out["shock"] = (
            (out["range"] > params.shock_multiple * out["baseline"])
            & (data["close"] > data["open"])
            & (out["close_location"] >= 0.75)
        ).fillna(False)
        out["contraction"] = (
            (out["range"] < out["range"].shift(1))
            & (out["range"].shift(1) < out["range"].shift(2))
        ).fillna(False)
        out["rising_floor"] = (
            (data["low"] > data["low"].shift(1))
            & (data["low"].shift(1) > data["low"].shift(2))
        ).fillna(False)
        out["rising_close"] = (
            (data["close"] > data["close"].shift(1))
            & (data["close"].shift(1) > data["close"].shift(2))
        ).fillna(False)
        out["ready"] = (
            out["baseline"].shift(2).notna()
            & np.isfinite(out["atr"])
            & (out["atr"] > 0.0)
            & np.isfinite(data["close"])
        )
        out["entry"] = (
            out["shock"].shift(2, fill_value=False)
            & out["contraction"] & out["rising_floor"]
            & out["rising_close"] & out["ready"]
        ).fillna(False)
        return out

    def generate_signals(
        self, data: pd.DataFrame, indicators: pd.DataFrame,
        ctx: StrategyContext, params: RangePulseParams,
    ) -> SignalFrame:
        closes = data["close"].to_numpy(dtype=float)
        atrs = indicators["atr"].to_numpy(dtype=float)
        entries = indicators["entry"].to_numpy(dtype=bool)
        ready = indicators["ready"].to_numpy(dtype=bool)
        signals = np.zeros(len(data), dtype=int)
        active = False
        age = 0
        high_water = np.nan
        stop = -np.inf
        for i in range(len(data)):
            if active:
                age += 1
                if not ready[i]:
                    active = False
                else:
                    # Entry is filled on this bar after yesterday's decision.
                    high_water = closes[i] if age == 1 else max(high_water, closes[i])
                    stop = max(stop, high_water - params.trail_atr * atrs[i])
                    if closes[i] <= stop or age >= 10:
                        active = False
                signals[i] = int(active)
                # An exit cannot reopen a position on the same decision bar.
                continue
            if ready[i] and entries[i]:
                active = True
                age = 0
                high_water = np.nan
                stop = -np.inf
                signals[i] = 1
        df = pd.DataFrame(index=data.index)
        df["signal"] = signals
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = 0.95
        return SignalFrame(data=df, signal_column="signal", size_column="size")
