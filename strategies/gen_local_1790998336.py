from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class FractionalPropagationParams:
    fractional_order: float = 0.45
    fractional_window: int = 48
    strength_window: int = 32
    efficiency_window: int = 8
    entry_z: float = 0.8
    min_efficiency: float = 0.28
    atr_window: int = 14
    breakeven_pct: float = 0.025
    trail_atr: float = 2.7
    max_hold_bars: int = 20


class GeneratedStrategy(BaseStrategy[FractionalPropagationParams]):
    strategy_id = "gen_local_1790998336"

    @classmethod
    def params_type(cls) -> type[FractionalPropagationParams]:
        return FractionalPropagationParams

    def warmup_bars(self, params: FractionalPropagationParams) -> int:
        return int(max(
            params.fractional_window + params.strength_window + params.efficiency_window + 2,
            params.atr_window + 1,
        ))

    def indicators(self, data: pd.DataFrame, params: FractionalPropagationParams) -> pd.DataFrame:
        close = data["close"].astype(float)
        width = max(2, int(params.fractional_window))
        weights = np.ones(width, dtype=float)
        for lag in range(1, width):
            weights[lag] = -weights[lag - 1] * (params.fractional_order - lag + 1) / lag
        fractional_close = close * weights[0]
        for lag in range(1, width):
            fractional_close = fractional_close + weights[lag] * close.shift(lag)
        center = fractional_close.rolling(params.strength_window, min_periods=params.strength_window).mean()
        dispersion = fractional_close.rolling(params.strength_window, min_periods=params.strength_window).std(ddof=0)
        displacement = (fractional_close - center) / dispersion.replace(0.0, np.nan)
        movement = fractional_close.diff()
        path_length = movement.abs().rolling(params.efficiency_window, min_periods=params.efficiency_window).sum()
        efficiency = fractional_close.diff(params.efficiency_window) / path_length.replace(0.0, np.nan)
        previous_close = close.shift(1)
        true_range = pd.concat([
            data["high"] - data["low"],
            (data["high"] - previous_close).abs(),
            (data["low"] - previous_close).abs(),
        ], axis=1).max(axis=1)
        atr = true_range.rolling(params.atr_window, min_periods=params.atr_window).mean()
        out = pd.DataFrame(index=data.index)
        out["fractional_close"] = fractional_close
        out["fractional_center"] = center
        out["fractional_dispersion"] = dispersion
        out["fractional_movement"] = movement
        out["fractional_path_length"] = path_length
        out["displacement"] = displacement
        out["efficiency"] = efficiency
        out["previous_close"] = previous_close
        out["true_range"] = true_range
        out["atr"] = atr
        valid = np.isfinite(displacement) & np.isfinite(efficiency) & np.isfinite(atr) & atr.gt(0)
        bullish = valid & displacement.ge(params.entry_z) & efficiency.ge(params.min_efficiency)
        bearish = valid & displacement.le(-params.entry_z) & efficiency.le(-params.min_efficiency)
        out["candidate"] = np.select([bullish, bearish], [1, -1], default=0).astype(int)
        candidate = out["candidate"]
        out["confirmed"] = candidate.where(candidate.eq(candidate.shift(1)) & candidate.ne(0), 0).astype(int)
        return out.replace([np.inf, -np.inf], np.nan)

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: FractionalPropagationParams) -> SignalFrame:
        closes = data["close"].to_numpy(dtype=float)
        opens = data["open"].to_numpy(dtype=float)
        atrs = indicators["atr"].to_numpy(dtype=float)
        entries = indicators["confirmed"].fillna(0).to_numpy(dtype=int)
        signals = np.zeros(len(data), dtype=int)
        direction = 0
        pending_entry = False
        entry_price = 0.0
        stop = -np.inf
        best_price = -np.inf
        armed = False
        held = 0
        warmup = self.warmup_bars(params)
        for i in range(len(data)):
            if pending_entry:
                previous_atr = atrs[i - 1] if i > 0 else np.nan
                if not np.isfinite(opens[i]) or opens[i] <= 0 or not np.isfinite(previous_atr) or previous_atr <= 0:
                    direction = 0
                else:
                    entry_price = opens[i]
                    best_price = direction * entry_price
                    stop = best_price - params.trail_atr * previous_atr
                    armed = False
                    held = 0
                pending_entry = False
            valid = np.isfinite(closes[i]) and closes[i] > 0 and np.isfinite(atrs[i]) and atrs[i] > 0
            if not valid:
                direction = 0
                continue
            if direction != 0:
                held += 1
                signed_close = direction * closes[i]
                signed_entry = direction * entry_price
                # Stops are monitored at the close; every exit fills next bar.
                if signed_close <= stop or held >= params.max_hold_bars:
                    direction = 0
                    continue
                best_price = max(best_price, signed_close)
                if best_price - signed_entry >= entry_price * params.breakeven_pct:
                    armed = True
                if armed:
                    # This coordinate only ratchets upward: a short price stop moves downward.
                    stop = max(stop, signed_entry, best_price - params.trail_atr * atrs[i])
                    if signed_close <= stop:
                        direction = 0
                        continue
                signals[i] = direction
            elif i >= warmup - 1 and entries[i] != 0:
                direction = int(entries[i])
                pending_entry = True
                signals[i] = direction
        df = pd.DataFrame(index=data.index)
        df["signal"] = signals
        df["size"] = 1.0
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")
