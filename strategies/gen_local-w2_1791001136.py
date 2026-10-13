from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class RecoveryParams:
    peak_window: int = 30
    fast_window: int = 2
    slow_window: int = 6
    atr_window: int = 14
    drawdown_min: float = 0.02
    drawdown_max: float = 0.12
    acceleration_min: float = 0.08
    breakeven_gain: float = 0.008
    initial_stop_atr: float = 2.0
    trail_atr: float = 1.7
    max_hold_bars: int = 4


class GeneratedStrategy(BaseStrategy[RecoveryParams]):
    strategy_id = "gen_local-w2_1791001136"

    @classmethod
    def params_type(cls) -> type[RecoveryParams]:
        return RecoveryParams

    def warmup_bars(self, params: RecoveryParams) -> int:
        return int(max(params.peak_window + 1, params.fast_window + 3,
                       params.slow_window + 3, params.atr_window + 2))

    def indicators(self, data: pd.DataFrame, params: RecoveryParams) -> pd.DataFrame:
        out = pd.DataFrame(index=data.index)
        close = data["close"].astype(float)
        out["previous_close"] = close.shift(1)
        out["peak"] = close.rolling(params.peak_window, min_periods=params.peak_window).max()
        out["drawdown"] = 1.0 - close / out["peak"].where(out["peak"] > 0)
        out["true_range"] = pd.concat([
            data["high"] - data["low"],
            (data["high"] - out["previous_close"]).abs(),
            (data["low"] - out["previous_close"]).abs(),
        ], axis=1).max(axis=1)
        out["atr"] = out["true_range"].rolling(params.atr_window, min_periods=params.atr_window).mean()
        out["fast_roc"] = close.pct_change(params.fast_window, fill_method=None)
        out["slow_roc"] = close.pct_change(params.slow_window, fill_method=None)
        out["fast_acceleration"] = out["fast_roc"].diff() / params.fast_window
        out["slow_acceleration"] = out["slow_roc"].diff() / params.slow_window
        out["atr_fraction"] = out["atr"] / close.where(close > 0)
        out["acceleration_spread"] = (
            out["fast_acceleration"] - out["slow_acceleration"]
        ) / out["atr_fraction"].where(out["atr_fraction"] > 0)
        out = out.replace([np.inf, -np.inf], np.nan)
        out["setup"] = (
            out["drawdown"].between(params.drawdown_min, params.drawdown_max)
            & (out["fast_roc"] > 0)
            & (out["slow_roc"] < 0)
            & (out["fast_acceleration"] > 0)
            & (out["acceleration_spread"] > params.acceleration_min)
            & (out["atr"] > 0)
        ).fillna(False)
        out["confirmed"] = out["setup"] & out["setup"].shift(1, fill_value=False)
        out.loc[np.arange(len(out)) < self.warmup_bars(params), "confirmed"] = False
        return out

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: RecoveryParams,
    ) -> SignalFrame:
        op = data["open"].to_numpy(dtype=float)
        hi = data["high"].to_numpy(dtype=float)
        lo = data["low"].to_numpy(dtype=float)
        cl = data["close"].to_numpy(dtype=float)
        atr = indicators["atr"].to_numpy(dtype=float)
        confirmed = indicators["confirmed"].fillna(False).to_numpy(dtype=bool)
        signal = np.zeros(len(data), dtype=int)
        active = False
        armed = False
        entry = stop = peak = 0.0
        entry_bar = -1
        hold_limit = min(5, max(3, int(params.max_hold_bars)))

        for i in range(len(data)):
            # Yesterday's entry decision becomes exposure at today's open.
            if not active and i > 0 and signal[i - 1] == 1:
                if np.isfinite(op[i]) and op[i] > 0 and np.isfinite(atr[i - 1]) and atr[i - 1] > 0:
                    active = True
                    armed = False
                    entry = op[i]
                    entry_bar = i
                    peak = entry
                    stop = max(0.0, entry - params.initial_stop_atr * atr[i - 1])

            if active:
                valid = np.isfinite([hi[i], lo[i], cl[i], atr[i]]).all() and atr[i] > 0
                exit_now = not valid
                if valid:
                    # Test the existing stop before ratcheting with today's high.
                    exit_now = lo[i] <= stop
                    peak = max(peak, hi[i])
                    armed = armed or peak >= entry * (1.0 + params.breakeven_gain)
                    if armed:
                        stop = max(stop, entry, peak - params.trail_atr * atr[i])
                        exit_now = exit_now or cl[i] <= stop
                    exit_now = exit_now or i - entry_bar + 1 >= hold_limit
                if exit_now:
                    active = False
                else:
                    signal[i] = 1
            elif confirmed[i] and i > 0 and np.isfinite(atr[i]) and atr[i] > 0:
                signal[i] = 1

        df = pd.DataFrame({"signal": signal, "size": 0.95}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")
