from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class AccelerationBreakoutParams:
    breakout_days: int = 21
    roc_days: int = 5
    acceleration_days: int = 3
    noise_days: int = 20
    efficiency_days: int = 20
    acceleration_threshold: float = 0.4
    minimum_efficiency: float = 0.35
    atr_days: int = 14
    initial_stop_atr: float = 2.5
    breakeven_gain: float = 0.025
    trail_atr: float = 2.5
    hold_bars: int = 18


class GeneratedStrategy(BaseStrategy[AccelerationBreakoutParams]):
    strategy_id = "gen_local_1791342091"

    @classmethod
    def params_type(cls) -> type[AccelerationBreakoutParams]:
        return AccelerationBreakoutParams

    def warmup_bars(self, params: AccelerationBreakoutParams) -> int:
        return int(max(
            200,
            params.breakout_days + 1,
            params.roc_days + params.acceleration_days + 1,
            params.noise_days + 1,
            params.efficiency_days + 1,
            params.atr_days + 1,
        ))

    def indicators(self, data: pd.DataFrame, params: AccelerationBreakoutParams) -> pd.DataFrame:
        lengths = (
            params.breakout_days, params.roc_days, params.acceleration_days,
            params.noise_days, params.efficiency_days, params.atr_days,
            params.hold_bars,
        )
        if any(value < 1 for value in lengths):
            raise ValueError("Lookbacks and holding period must be positive")
        thresholds = (
            params.acceleration_threshold, params.minimum_efficiency,
            params.initial_stop_atr, params.breakeven_gain, params.trail_atr,
        )
        if not all(np.isfinite(value) for value in thresholds):
            raise ValueError("Thresholds must be finite")
        if (params.acceleration_threshold < 0 or
                not 0 <= params.minimum_efficiency <= 1 or
                params.initial_stop_atr <= 0 or params.breakeven_gain <= 0 or
                params.trail_atr <= 0):
            raise ValueError("Invalid entry or stop thresholds")

        out = pd.DataFrame(index=data.index)
        close = data["close"]
        out["previous_close"] = close.shift(1)
        out["daily_return"] = close.div(out["previous_close"].replace(0, np.nan)).sub(1)
        out["regime_ma"] = close.rolling(200, min_periods=200).mean()
        out["breakout_ceiling"] = data["high"].rolling(
            params.breakout_days, min_periods=params.breakout_days
        ).max().shift(1)
        out["roc"] = close.div(close.shift(params.roc_days).replace(0, np.nan)).sub(1)
        out["roc_acceleration"] = out["roc"].diff(params.acceleration_days)
        out["return_noise"] = out["daily_return"].rolling(
            params.noise_days, min_periods=params.noise_days
        ).std(ddof=0)
        out["normalized_acceleration"] = out["roc_acceleration"].div(
            (out["return_noise"] * np.sqrt(params.roc_days)).replace(0, np.nan)
        )
        out["price_change"] = close.diff()
        out["path_length"] = out["price_change"].abs().rolling(
            params.efficiency_days, min_periods=params.efficiency_days
        ).sum()
        out["net_displacement"] = close.diff(params.efficiency_days)
        out["efficiency"] = out["net_displacement"].div(
            out["path_length"].replace(0, np.nan)
        )
        out["true_range"] = pd.concat([
            data["high"] - data["low"],
            (data["high"] - out["previous_close"]).abs(),
            (data["low"] - out["previous_close"]).abs(),
        ], axis=1).max(axis=1)
        out["atr"] = out["true_range"].rolling(
            params.atr_days, min_periods=params.atr_days
        ).mean()
        return out.replace([np.inf, -np.inf], np.nan)

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: AccelerationBreakoutParams,
    ) -> SignalFrame:
        ind = indicators
        close = data["close"].to_numpy(dtype=float)
        opening = data["open"].to_numpy(dtype=float)
        atr = ind["atr"].to_numpy(dtype=float)
        required = [
            "regime_ma", "breakout_ceiling", "roc",
            "normalized_acceleration", "efficiency", "atr",
        ]
        ready = ind[required].notna().all(axis=1)
        entry_condition = (
            ready & (data["close"] > ind["regime_ma"])
            & (data["close"] > ind["breakout_ceiling"])
            & (ind["roc"] > 0)
            & (ind["normalized_acceleration"] > params.acceleration_threshold)
            & (ind["efficiency"] >= params.minimum_efficiency)
            & (ind["atr"] > 0)
        ).fillna(False).to_numpy(dtype=bool)

        desired = np.zeros(len(data), dtype=int)
        active = False
        pending_entry = False
        breakeven_active = False
        entry_price = 0.0
        stop = 0.0
        age = 0

        # Stops are evaluated at the close; every exit executes next bar.
        for i in range(len(data)):
            if pending_entry:
                pending_entry = False
                if (i > 0 and np.isfinite(opening[i]) and opening[i] > 0
                        and np.isfinite(atr[i - 1]) and atr[i - 1] > 0):
                    active = True
                    entry_price = opening[i]
                    stop = entry_price - params.initial_stop_atr * atr[i - 1]
                    breakeven_active = False
                    age = 0
                else:
                    continue

            if active:
                age += 1
                if not np.isfinite(close[i]) or close[i] <= 0:
                    active = False
                    continue
                if close[i] <= stop or age >= params.hold_bars:
                    active = False
                    continue
                if close[i] >= entry_price * (1.0 + params.breakeven_gain):
                    breakeven_active = True
                if breakeven_active:
                    stop = max(stop, entry_price)
                    if np.isfinite(atr[i]) and atr[i] > 0:
                        stop = max(stop, close[i] - params.trail_atr * atr[i])
                    if close[i] <= stop:
                        active = False
                        continue
                desired[i] = 1
            elif entry_condition[i]:
                desired[i] = 1
                pending_entry = True

        df = pd.DataFrame(index=data.index)
        df["signal"] = desired
        df["size"] = 0.95
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")
