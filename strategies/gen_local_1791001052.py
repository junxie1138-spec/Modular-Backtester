from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class FractionalMemoryParams:
    fd_window: int = 32
    fd_order: float = 0.45
    strength_window: int = 8
    atr_window: int = 14
    min_strength: float = 0.6
    min_efficiency: float = 0.55
    stop_atr: float = 1.25
    hold_bars: int = 2


class GeneratedStrategy(BaseStrategy[FractionalMemoryParams]):
    strategy_id = "gen_local_1791001052"

    @classmethod
    def params_type(cls) -> type[FractionalMemoryParams]:
        return FractionalMemoryParams

    def warmup_bars(self, params: FractionalMemoryParams) -> int:
        return int(max(max(2, params.fd_window) + max(2, params.strength_window) + 1, max(2, params.atr_window) + 1))

    def indicators(self, data: pd.DataFrame, params: FractionalMemoryParams) -> pd.DataFrame:
        window = max(2, int(params.fd_window))
        strength_window = max(2, int(params.strength_window))
        atr_window = max(2, int(params.atr_window))
        order = float(np.clip(params.fd_order, 0.01, 0.99))
        close = data["close"].astype(float)
        weights = np.empty(window, dtype=float)
        weights[0] = 1.0
        for lag in range(1, window):
            weights[lag] = -weights[lag - 1] * (order - lag + 1.0) / lag
        fractional_close = close * weights[0]
        for lag in range(1, window):
            fractional_close = fractional_close + weights[lag] * close.shift(lag)
        previous_close = close.shift(1)
        true_range = pd.concat([
            data["high"] - data["low"],
            (data["high"] - previous_close).abs(),
            (data["low"] - previous_close).abs(),
        ], axis=1).max(axis=1)
        atr = true_range.rolling(atr_window, min_periods=atr_window).mean()
        baseline = fractional_close.rolling(strength_window, min_periods=strength_window).mean()
        increment = fractional_close.diff()
        net_increment = increment.rolling(strength_window, min_periods=strength_window).sum()
        total_increment = increment.abs().rolling(strength_window, min_periods=strength_window).sum()
        result = pd.DataFrame(index=data.index)
        result["fractional_close"] = fractional_close
        result["fractional_baseline"] = baseline
        result["fractional_increment"] = increment
        result["net_increment"] = net_increment
        result["total_increment"] = total_increment
        result["true_range"] = true_range
        result["atr"] = atr
        result["strength"] = ((fractional_close - baseline) / atr.where(atr > 0)).replace([np.inf, -np.inf], np.nan)
        result["efficiency"] = (net_increment / total_increment.where(total_increment > 0)).replace([np.inf, -np.inf], np.nan)
        return result

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: FractionalMemoryParams) -> SignalFrame:
        strength = indicators["strength"]
        efficiency = indicators["efficiency"]
        threshold = max(0.0, float(params.min_strength))
        efficiency_threshold = float(np.clip(params.min_efficiency, 0.0, 1.0))
        valid = indicators[["strength", "efficiency", "atr"]].notna().all(axis=1) & (indicators["atr"] > 0)
        long_candidate = valid & (strength > threshold) & (efficiency > efficiency_threshold)
        short_candidate = valid & (strength < -threshold) & (efficiency < -efficiency_threshold)
        long_confirmed = long_candidate & long_candidate.shift(1, fill_value=False)
        short_confirmed = short_candidate & short_candidate.shift(1, fill_value=False)
        entries = np.select([long_confirmed, short_confirmed], [1, -1], default=0)
        closes = data["close"].to_numpy(dtype=float)
        opens = data["open"].to_numpy(dtype=float)
        atrs = indicators["atr"].to_numpy(dtype=float)
        signals = np.zeros(len(data), dtype=int)
        position = 0
        entry_price = np.nan
        stop_distance = np.nan
        age = 0
        holding_limit = min(2, max(1, int(params.hold_bars)))
        stop_multiple = max(0.01, float(params.stop_atr))
        for i in range(len(data)):
            if position != 0:
                # The previous close's decision fills at this bar's open.
                if age == 0:
                    entry_price = opens[i]
                age += 1
                prices_valid = np.isfinite(entry_price) and np.isfinite(closes[i])
                stopped = prices_valid and (
                    (position == 1 and closes[i] < entry_price - stop_distance)
                    or (position == -1 and closes[i] > entry_price + stop_distance)
                )
                if not prices_valid or stopped or age >= holding_limit:
                    position = 0
                    age = 0
                    entry_price = np.nan
                    stop_distance = np.nan
            elif entries[i] != 0 and np.isfinite(closes[i]) and np.isfinite(atrs[i]) and atrs[i] > 0:
                position = int(entries[i])
                entry_price = np.nan
                stop_distance = stop_multiple * atrs[i]
                age = 0
            signals[i] = position
        df = pd.DataFrame(index=data.index)
        df["signal"] = signals
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = 1.0
        return SignalFrame(data=df, signal_column="signal", size_column="size")
