from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class StrategyParams:
    frac_order: float = 0.4
    frac_window: int = 20
    z_window: int = 40
    z_entry: float = 1.5
    wick_threshold: float = 0.45
    atr_window: int = 14
    trail_atr: float = 1.2
    hold_bars: int = 2


class GeneratedStrategy(BaseStrategy[StrategyParams]):
    strategy_id = "gen_local_1791232476"

    @classmethod
    def params_type(cls) -> type[StrategyParams]:
        return StrategyParams

    def warmup_bars(self, params: StrategyParams) -> int:
        return int(max(params.frac_window + params.z_window, params.atr_window + 1))

    def indicators(self, data: pd.DataFrame, params: StrategyParams) -> pd.DataFrame:
        out = pd.DataFrame(index=data.index)
        close = data["close"].astype(float)
        fractional = close.copy()
        weight = 1.0
        for lag in range(1, params.frac_window):
            weight *= (lag - 1 - params.frac_order) / lag
            fractional = fractional + weight * close.shift(lag)
        out["fractional_close"] = fractional
        historical = fractional.shift(1).rolling(params.z_window, min_periods=params.z_window)
        out["fractional_mean"] = historical.mean()
        out["fractional_std"] = historical.std(ddof=0)
        out["fractional_z"] = (
            (fractional - out["fractional_mean"])
            / out["fractional_std"].replace(0.0, np.nan)
        )
        out["range"] = data["high"] - data["low"]
        out["lower_wick"] = (
            data[["open", "close"]].min(axis=1) - data["low"]
        ).clip(lower=0.0)
        out["wick_fraction"] = (
            out["lower_wick"] / out["range"].replace(0.0, np.nan)
        )
        previous = close.shift(1)
        out["true_range"] = pd.concat(
            [out["range"], (data["high"] - previous).abs(),
             (data["low"] - previous).abs()], axis=1
        ).max(axis=1)
        out["atr"] = out["true_range"].rolling(
            params.atr_window, min_periods=params.atr_window
        ).mean()
        return out.replace([np.inf, -np.inf], np.nan)

    def generate_signals(
        self, data: pd.DataFrame, indicators: pd.DataFrame,
        ctx: StrategyContext, params: StrategyParams,
    ) -> SignalFrame:
        entry = (
            (indicators["fractional_z"] <= -params.z_entry)
            & (indicators["wick_fraction"] >= params.wick_threshold)
        ).fillna(False).to_numpy(dtype=bool)
        closes = data["close"].to_numpy(dtype=float)
        atrs = indicators["atr"].to_numpy(dtype=float)
        signals = np.zeros(len(data), dtype=int)
        active = False
        held = 0
        high_water = -np.inf
        stop = -np.inf
        for i in range(len(data)):
            close, atr = closes[i], atrs[i]
            valid = np.isfinite(close) and np.isfinite(atr) and atr > 0.0
            if active:
                held += 1
                if not valid:
                    active = False
                    continue
                # The first active bar is the session after the entry decision.
                high_water = max(high_water, close)
                stop = max(stop, high_water - params.trail_atr * atr)
                if close <= stop or held >= params.hold_bars:
                    active = False
                    continue
                signals[i] = 1
            elif valid and entry[i]:
                active = True
                held = 0
                high_water = -np.inf
                stop = -np.inf
                signals[i] = 1
        df = pd.DataFrame({"signal": signals, "size": 0.95}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")
