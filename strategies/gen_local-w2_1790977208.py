from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class FloorStreakParams:
    streak_length: int = 3
    efficiency_window: int = 10
    min_efficiency: float = 0.30
    atr_window: int = 14
    breakeven_pct: float = 0.004
    trail_atr: float = 1.2
    max_hold_bars: int = 2


class GeneratedStrategy(BaseStrategy[FloorStreakParams]):
    strategy_id = "gen_local-w2_1790977208"

    @classmethod
    def params_type(cls) -> type[FloorStreakParams]:
        return FloorStreakParams

    def warmup_bars(self, params: FloorStreakParams) -> int:
        return int(max(200, params.atr_window + 1, params.efficiency_window + 1, params.streak_length + 1))

    def indicators(self, data: pd.DataFrame, params: FloorStreakParams) -> pd.DataFrame:
        if params.streak_length < 1 or params.efficiency_window < 1 or params.atr_window < 1:
            raise ValueError("Lookbacks must be positive")
        if not 0.0 <= params.min_efficiency <= 1.0:
            raise ValueError("min_efficiency must be between zero and one")
        if not np.isfinite(params.trail_atr) or params.trail_atr <= 0:
            raise ValueError("trail_atr must be positive and finite")
        if not np.isfinite(params.breakeven_pct) or params.breakeven_pct <= 0:
            raise ValueError("breakeven_pct must be positive and finite")
        if params.max_hold_bars not in (1, 2):
            raise ValueError("max_hold_bars must be one or two")

        out = pd.DataFrame(index=data.index)
        close = data["close"]
        previous_close = close.shift(1)
        out["ma200"] = close.rolling(200, min_periods=200).mean()
        out["rising_low"] = data["low"].gt(data["low"].shift(1))
        groups = (~out["rising_low"]).cumsum()
        out["floor_streak"] = out["rising_low"].astype(int).groupby(groups).cumsum()
        out["close_change"] = close.diff()
        out["path_length"] = out["close_change"].abs().rolling(params.efficiency_window, min_periods=params.efficiency_window).sum()
        out["displacement"] = close.diff(params.efficiency_window)
        out["efficiency"] = out["displacement"].div(out["path_length"].replace(0.0, np.nan))
        out["true_range"] = pd.concat([
            data["high"] - data["low"],
            (data["high"] - previous_close).abs(),
            (data["low"] - previous_close).abs(),
        ], axis=1).max(axis=1)
        out["atr"] = out["true_range"].rolling(params.atr_window, min_periods=params.atr_window).mean()
        out["eligible"] = (
            out["floor_streak"].eq(params.streak_length)
            & close.gt(out["ma200"])
            & out["efficiency"].ge(params.min_efficiency)
            & out["atr"].gt(0.0)
            & np.isfinite(out["atr"])
            & np.isfinite(close)
        ).fillna(False)
        return out

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: FloorStreakParams) -> SignalFrame:
        df = pd.DataFrame(index=data.index)
        df["signal"] = 0
        df["size"] = 0.95
        signals = np.zeros(len(data), dtype=int)
        opens = data["open"].to_numpy(dtype=float)
        highs = data["high"].to_numpy(dtype=float)
        lows = data["low"].to_numpy(dtype=float)
        closes = data["close"].to_numpy(dtype=float)
        atrs = indicators["atr"].to_numpy(dtype=float)
        eligible = indicators["eligible"].to_numpy(dtype=bool)
        active = False
        enter_next = False
        armed = False
        entry = 0.0
        stop = 0.0
        peak = 0.0
        held = 0

        for i in range(len(data)):
            if enter_next:
                enter_next = False
                if i > 0 and np.isfinite(opens[i]) and opens[i] > 0 and np.isfinite(atrs[i - 1]):
                    active = True
                    entry = opens[i]
                    peak = entry
                    stop = entry - params.trail_atr * atrs[i - 1]
                    armed = False
                    held = 0

            if active:
                held += 1
                # Test the previously established stop before raising it.
                exit_now = bool(np.isfinite(lows[i]) and lows[i] <= stop)
                if np.isfinite(highs[i]):
                    peak = max(peak, highs[i])
                    armed = armed or peak >= entry * (1.0 + params.breakeven_pct)
                if armed:
                    stop = max(stop, entry)
                    if np.isfinite(atrs[i]) and atrs[i] > 0:
                        stop = max(stop, peak - params.trail_atr * atrs[i])
                # A newly raised stop is evaluated at this close; execution
                # still occurs at the following open through the final shift.
                exit_now = exit_now or bool(np.isfinite(closes[i]) and closes[i] <= stop)
                exit_now = exit_now or held >= params.max_hold_bars
                if exit_now:
                    active = False
                    continue
                signals[i] = 1
            elif eligible[i]:
                signals[i] = 1
                enter_next = True

        df["signal"] = signals
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")
