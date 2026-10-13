from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class BreakoutParams:
    roc_bars: int = 3
    frontier_bars: int = 12
    atr_bars: int = 10
    min_acceleration: float = 0.15
    trail_atr: float = 1.6
    max_hold: int = 4


class GeneratedStrategy(BaseStrategy[BreakoutParams]):
    strategy_id = "gen_local-w2_1790990675"

    @classmethod
    def params_type(cls) -> type[BreakoutParams]:
        return BreakoutParams

    def warmup_bars(self, params: BreakoutParams) -> int:
        return int(max(params.roc_bars + params.frontier_bars + 2,
                       params.frontier_bars + 1, params.atr_bars + 1))

    def indicators(self, data: pd.DataFrame, params: BreakoutParams) -> pd.DataFrame:
        close = data["close"]
        previous = close.shift(1)
        tr = pd.concat([
            data["high"] - data["low"],
            (data["high"] - previous).abs(),
            (data["low"] - previous).abs()
        ], axis=1).max(axis=1)
        atr = tr.rolling(params.atr_bars, min_periods=params.atr_bars).mean()
        roc = close.pct_change(params.roc_bars, fill_method=None)
        acceleration = roc.diff() * close / atr.replace(0.0, np.nan)
        out = pd.DataFrame(index=data.index)
        out["atr"] = atr
        out["roc"] = roc
        out["acceleration"] = acceleration
        out["acceleration_frontier"] = acceleration.shift(1).rolling(
            params.frontier_bars, min_periods=params.frontier_bars
        ).max()
        out["price_frontier"] = data["high"].shift(1).rolling(
            params.frontier_bars, min_periods=params.frontier_bars
        ).max()
        return out.replace([np.inf, -np.inf], np.nan)

    def generate_signals(
        self, data: pd.DataFrame, indicators: pd.DataFrame,
        ctx: StrategyContext, params: BreakoutParams
    ) -> SignalFrame:
        ready = indicators.notna().all(axis=1)
        momentum_breakout = (
            (indicators["roc"] > 0.0)
            & (indicators["acceleration"] > params.min_acceleration)
            & (indicators["acceleration"] > indicators["acceleration_frontier"])
        )
        price_breakout = data["close"] > indicators["price_frontier"]
        entries = (ready & momentum_breakout & price_breakout).fillna(False)
        signals = np.zeros(len(data), dtype=int)
        active = False
        high_water = 0.0
        stop = 0.0
        age = 0
        for i in range(len(data)):
            close = float(data["close"].iloc[i])
            atr = float(indicators["atr"].iloc[i])
            valid = bool(ready.iloc[i]) and np.isfinite(close) and atr > 0.0
            if active:
                age += 1
                if not valid:
                    active = False
                else:
                    high_water = max(high_water, close)
                    stop = max(stop, high_water - params.trail_atr * atr)
                    if close <= stop or age >= params.max_hold:
                        active = False
                # Exits cannot immediately reopen on the same decision bar.
            elif valid and bool(entries.iloc[i]):
                active = True
                age = 0
                high_water = close
                stop = high_water - params.trail_atr * atr
            signals[i] = int(active)
        df = pd.DataFrame({"signal": signals, "size": 0.95}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")
