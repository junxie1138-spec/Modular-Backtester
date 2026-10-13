from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class ResidencyParams:
    channel_bars: int = 20
    streak_bars: int = 3
    atr_bars: int = 14
    lower_position: float = 0.50
    upper_position: float = 0.85
    max_drift_atr: float = 1.0
    trail_atr: float = 2.0
    max_hold_bars: int = 10


class GeneratedStrategy(BaseStrategy[ResidencyParams]):
    strategy_id = "gen_local-w2_1790991093"

    @classmethod
    def params_type(cls) -> type[ResidencyParams]:
        return ResidencyParams

    def warmup_bars(self, params: ResidencyParams) -> int:
        return int(max(params.channel_bars + 1, params.atr_bars + 1, params.streak_bars + 1) + params.streak_bars + 2)

    def indicators(self, data: pd.DataFrame, params: ResidencyParams) -> pd.DataFrame:
        result = pd.DataFrame(index=data.index)
        close = data["close"].astype(float)
        result["channel_high"] = data["high"].rolling(params.channel_bars, min_periods=params.channel_bars).max().shift(1)
        result["channel_low"] = data["low"].rolling(params.channel_bars, min_periods=params.channel_bars).min().shift(1)
        result["channel_width"] = result["channel_high"] - result["channel_low"]
        denominator = result["channel_width"].where(result["channel_width"] > 0.0)
        result["position"] = ((close - result["channel_low"]) / denominator).replace([np.inf, -np.inf], np.nan)
        previous_close = close.shift(1)
        result["true_range"] = pd.concat([
            data["high"] - data["low"],
            (data["high"] - previous_close).abs(),
            (data["low"] - previous_close).abs(),
        ], axis=1).max(axis=1).where(previous_close.notna())
        result["atr"] = result["true_range"].rolling(params.atr_bars, min_periods=params.atr_bars).mean()
        result["drift_atr"] = (close.diff(params.streak_bars).abs() / result["atr"].where(result["atr"] > 0.0)).replace([np.inf, -np.inf], np.nan)
        resident = (
            result["position"].ge(params.lower_position)
            & result["position"].le(params.upper_position)
            & result["atr"].gt(0.0)
            & np.isfinite(close)
        ).fillna(False)
        result["resident"] = resident.astype(int)
        result["residency_streak"] = resident.astype(int).groupby((~resident).cumsum()).cumsum()
        result["setup"] = (
            result["residency_streak"].ge(params.streak_bars)
            & result["drift_atr"].le(params.max_drift_atr)
        ).fillna(False)
        result["confirmed"] = result["setup"] & result["setup"].shift(1, fill_value=False)
        result["entry_event"] = result["confirmed"] & ~result["confirmed"].shift(1, fill_value=False)
        return result

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: ResidencyParams) -> SignalFrame:
        closes = data["close"].to_numpy(dtype=float)
        atrs = indicators["atr"].to_numpy(dtype=float)
        entries = indicators["entry_event"].fillna(False).to_numpy(dtype=bool)
        signals = np.zeros(len(data), dtype=int)
        active = False
        high_water = 0.0
        stop = 0.0
        entry_bar = -1
        warmup = self.warmup_bars(params)

        for i in range(len(data)):
            close = closes[i]
            atr = atrs[i]
            valid = np.isfinite(close) and np.isfinite(atr) and close > 0.0 and atr > 0.0
            if not valid:
                active = False
                continue
            if active:
                high_water = max(high_water, close)
                stop = max(stop, high_water - params.trail_atr * atr)
                if close <= stop or i - entry_bar >= params.max_hold_bars:
                    active = False
            elif i >= warmup - 1 and entries[i]:
                active = True
                entry_bar = i
                high_water = close
                stop = close - params.trail_atr * atr
            signals[i] = int(active)

        df = pd.DataFrame(index=data.index)
        df["signal"] = signals
        df["size"] = 1.0
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")
