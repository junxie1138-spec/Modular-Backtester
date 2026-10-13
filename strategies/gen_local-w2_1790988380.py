from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class ExtensionRejectionParams:
    streak_bars: int = 4
    pressure_bars: int = 6
    pressure_threshold: float = 0.20
    holding_bars: int = 18


class GeneratedStrategy(BaseStrategy[ExtensionRejectionParams]):
    strategy_id = "gen_local-w2_1790988380"

    @classmethod
    def params_type(cls) -> type[ExtensionRejectionParams]:
        return ExtensionRejectionParams

    def warmup_bars(self, params: ExtensionRejectionParams) -> int:
        return int(max(params.streak_bars + 1, params.pressure_bars))

    def indicators(self, data: pd.DataFrame, params: ExtensionRejectionParams) -> pd.DataFrame:
        if params.streak_bars < 1 or params.pressure_bars < 1 or params.holding_bars < 1:
            raise ValueError("Bar counts must be positive")
        if not 0.0 < params.pressure_threshold <= 1.0:
            raise ValueError("pressure_threshold must be in (0, 1]")

        lower = data["low"].diff().lt(0)
        higher = data["high"].diff().gt(0)
        lower_streak = lower.astype(int).groupby((~lower).cumsum()).cumsum()
        higher_streak = higher.astype(int).groupby((~higher).cumsum()).cumsum()

        spread = (data["high"] - data["low"]).replace(0.0, np.nan)
        location = ((2.0 * data["close"] - data["high"] - data["low"]) / spread).clip(-1.0, 1.0)
        volume = data["volume"].where(data["volume"] > 0.0)
        valid = location.notna() & volume.notna()
        weight = volume.where(valid)
        numerator = (location * weight).rolling(params.pressure_bars, min_periods=params.pressure_bars).sum()
        denominator = weight.rolling(params.pressure_bars, min_periods=params.pressure_bars).sum().replace(0.0, np.nan)
        pressure = (numerator / denominator).replace([np.inf, -np.inf], np.nan)

        return pd.DataFrame({
            "lower_streak": lower_streak.clip(upper=params.streak_bars).fillna(0).astype(int),
            "higher_streak": higher_streak.clip(upper=params.streak_bars).fillna(0).astype(int),
            "pressure": pressure.fillna(0.0),
            "pressure_ready": pressure.notna(),
        }, index=data.index)

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: ExtensionRejectionParams) -> SignalFrame:
        ready = indicators["pressure_ready"] & (np.arange(len(data)) >= self.warmup_bars(params) - 1)
        buy = ready & indicators["lower_streak"].ge(params.streak_bars) & indicators["pressure"].ge(params.pressure_threshold)
        sell = ready & indicators["higher_streak"].ge(params.streak_bars) & indicators["pressure"].le(-params.pressure_threshold)
        entries = np.select([buy & ~sell, sell & ~buy], [1, -1], default=0).astype(int)

        signals = np.zeros(len(data), dtype=int)
        position = 0
        exit_bar = -1
        for i in range(len(data)):
            if position != 0 and i == exit_bar:
                position = 0
                continue
            if position == 0 and entries[i] != 0:
                position = int(entries[i])
                exit_bar = i + params.holding_bars
            signals[i] = position

        df = pd.DataFrame({"signal": signals, "size": 1.0}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")
