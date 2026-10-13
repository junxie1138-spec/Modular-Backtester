from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class DrawdownDebtParams:
    peak_window: int = 63
    debt_window: int = 20
    atr_window: int = 14
    min_mean_depth: float = 0.035
    remaining_fraction: float = 0.55
    min_entry_depth: float = 0.012
    breakeven_gain: float = 0.025
    initial_stop_atr: float = 2.5
    trail_atr: float = 2.5
    max_hold_bars: int = 20


class GeneratedStrategy(BaseStrategy[DrawdownDebtParams]):
    strategy_id = "gen_local_1790999278"

    @classmethod
    def params_type(cls):
        return DrawdownDebtParams

    def warmup_bars(self, params: DrawdownDebtParams) -> int:
        return int(max(params.peak_window + params.debt_window + 1,
                       params.atr_window + 1))

    def indicators(self, data: pd.DataFrame,
                   params: DrawdownDebtParams) -> pd.DataFrame:
        out = pd.DataFrame(index=data.index)
        close = data["close"].where(data["close"] > 0)
        out["peak"] = close.rolling(
            params.peak_window, min_periods=params.peak_window
        ).max()
        out["depth"] = (1.0 - close / out["peak"]).clip(lower=0.0)
        out["mean_depth"] = out["depth"].rolling(
            params.debt_window, min_periods=params.debt_window
        ).mean()
        out["anchor_min"] = out["peak"].rolling(
            params.debt_window, min_periods=params.debt_window
        ).min()
        out["anchor_max"] = out["peak"].rolling(
            params.debt_window, min_periods=params.debt_window
        ).max()
        previous_close = close.shift(1)
        out["true_range"] = pd.concat([
            data["high"] - data["low"],
            (data["high"] - previous_close).abs(),
            (data["low"] - previous_close).abs()
        ], axis=1).max(axis=1)
        out["atr"] = out["true_range"].rolling(
            params.atr_window, min_periods=params.atr_window
        ).mean()
        out["eligible"] = (
            out["mean_depth"].ge(params.min_mean_depth)
            & out["depth"].ge(params.min_entry_depth)
            & out["depth"].le(
                params.remaining_fraction * out["mean_depth"]
            )
            & out["anchor_min"].eq(out["anchor_max"])
            & out["atr"].gt(0)
            & close.notna()
        ).fillna(False)
        out["confirmed"] = (
            out["eligible"]
            & out["eligible"].shift(1, fill_value=False)
        )
        out["entry_event"] = (
            out["confirmed"]
            & ~out["confirmed"].shift(1, fill_value=False)
        )
        return out

    def generate_signals(self, data: pd.DataFrame,
                         indicators: pd.DataFrame,
                         ctx: StrategyContext,
                         params: DrawdownDebtParams) -> SignalFrame:
        decisions = np.zeros(len(data), dtype=int)
        close = data["close"].to_numpy(dtype=float)
        opens = data["open"].to_numpy(dtype=float)
        atr = indicators["atr"].to_numpy(dtype=float)
        entries = indicators["entry_event"].to_numpy(dtype=bool)
        active = False
        pending = False
        armed = False
        entry = np.nan
        stop = np.nan
        highest_close = np.nan
        entry_bar = -1
        warmup = self.warmup_bars(params)

        for i in range(len(data)):
            if pending:
                # Yesterday's entry decision fills at today's open.
                pending = False
                if (np.isfinite(opens[i]) and opens[i] > 0
                        and i > 0 and np.isfinite(atr[i - 1])
                        and atr[i - 1] > 0):
                    entry = opens[i]
                    entry_bar = i
                    stop = entry - params.initial_stop_atr * atr[i - 1]
                    highest_close = entry
                    armed = False
                    active = True

            if active:
                if not (np.isfinite(close[i]) and close[i] > 0
                        and np.isfinite(atr[i]) and atr[i] > 0):
                    active = False
                else:
                    highest_close = max(highest_close, close[i])
                    # All stop decisions use closes and execute next bar.
                    if close[i] >= entry * (1.0 + params.breakeven_gain):
                        armed = True
                    if armed:
                        stop = max(stop, entry,
                                   highest_close - params.trail_atr * atr[i])
                    if (close[i] <= stop
                            or i - entry_bar + 1 >= params.max_hold_bars):
                        active = False
                decisions[i] = int(active)
            elif (i >= warmup - 1 and entries[i]
                  and np.isfinite(close[i]) and close[i] > 0
                  and np.isfinite(atr[i]) and atr[i] > 0):
                pending = True
                decisions[i] = 1

        df = pd.DataFrame(index=data.index)
        df["signal"] = decisions
        df["size"] = 1.0
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")
