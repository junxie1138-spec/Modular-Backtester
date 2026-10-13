from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class PressureParams:
    pressure_window: int = 12
    volume_window: int = 20
    atr_window: int = 14
    pressure_threshold: float = 0.12
    volume_floor: float = 1.0
    trail_atr: float = 2.3
    max_hold_bars: int = 10


class GeneratedStrategy(BaseStrategy[PressureParams]):
    strategy_id = "gen_local_1790988508"

    @classmethod
    def params_type(cls) -> type[PressureParams]:
        return PressureParams

    def warmup_bars(self, params: PressureParams) -> int:
        return int(max(max(1, params.volume_window), max(1, params.atr_window)) + max(2, params.pressure_window) + 2)

    def indicators(self, data: pd.DataFrame, params: PressureParams) -> pd.DataFrame:
        pressure_window = max(2, int(params.pressure_window))
        volume_window = max(1, int(params.volume_window))
        atr_window = max(1, int(params.atr_window))
        out = pd.DataFrame(index=data.index)
        out["previous_close"] = data["close"].shift(1)
        out["close_change"] = data["close"].diff()
        out["true_range"] = pd.concat([
            data["high"] - data["low"],
            (data["high"] - out["previous_close"]).abs(),
            (data["low"] - out["previous_close"]).abs(),
        ], axis=1).max(axis=1)
        out["atr"] = out["true_range"].rolling(atr_window, min_periods=atr_window).mean()
        out["volume_baseline"] = data["volume"].rolling(volume_window, min_periods=volume_window).mean().shift(1)
        out["volume_ratio"] = data["volume"] / out["volume_baseline"].where(out["volume_baseline"] > 0)
        out["normalized_move"] = (out["close_change"] / out["atr"].where(out["atr"] > 0)).clip(-2.0, 2.0)
        out["contribution"] = out["normalized_move"] * out["volume_ratio"]
        out["pressure_sum"] = out["contribution"].rolling(pressure_window, min_periods=pressure_window).sum()
        out["largest_positive"] = out["contribution"].clip(lower=0.0).rolling(pressure_window, min_periods=pressure_window).max()
        out["residual_pressure"] = (out["pressure_sum"] - out["largest_positive"]) / float(pressure_window - 1)
        return out.replace([np.inf, -np.inf], np.nan)

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: PressureParams) -> SignalFrame:
        qualifies = (
            indicators["residual_pressure"].ge(params.pressure_threshold)
            & indicators["volume_ratio"].ge(params.volume_floor)
            & indicators["close_change"].gt(0.0)
            & indicators["atr"].gt(0.0)
        ).fillna(False)
        confirmed = (qualifies & qualifies.shift(1, fill_value=False)).to_numpy(dtype=bool)
        closes = data["close"].to_numpy(dtype=float)
        atrs = indicators["atr"].to_numpy(dtype=float)
        positions = np.zeros(len(data), dtype=np.int64)
        active = False
        entry_decision = -1
        high_water = -np.inf
        stop = -np.inf
        trail_multiple = max(0.01, float(params.trail_atr))
        max_hold = max(1, int(params.max_hold_bars))

        for i in range(len(data)):
            close = closes[i]
            atr = atrs[i]
            valid = np.isfinite(close) and close > 0.0 and np.isfinite(atr) and atr > 0.0
            if active:
                if not valid:
                    active = False
                else:
                    # The first tracked close belongs to the actual fill bar.
                    high_water = max(high_water, close)
                    # ATR expansion cannot lower a previously established stop.
                    stop = max(stop, high_water - trail_multiple * atr)
                    if close <= stop or i - entry_decision >= max_hold:
                        active = False
            elif valid and confirmed[i]:
                active = True
                entry_decision = i
                high_water = -np.inf
                stop = -np.inf
            positions[i] = int(active)

        df = pd.DataFrame(index=data.index)
        df["signal"] = positions
        df["size"] = 1.0
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")
