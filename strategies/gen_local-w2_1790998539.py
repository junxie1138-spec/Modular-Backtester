from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class GapPressureParams:
    streak_bars: int = 3
    min_gap: float = 0.0005
    pressure_bars: int = 5
    pressure_threshold: float = 0.20
    hold_bars: int = 8


class GeneratedStrategy(BaseStrategy[GapPressureParams]):
    strategy_id = "gen_local-w2_1790998539"

    @classmethod
    def params_type(cls) -> type[GapPressureParams]:
        return GapPressureParams

    def warmup_bars(self, params: GapPressureParams) -> int:
        return int(max(params.streak_bars + 1, params.pressure_bars))

    def indicators(self, data: pd.DataFrame, params: GapPressureParams) -> pd.DataFrame:
        out = pd.DataFrame(index=data.index)
        prior_close = data["close"].shift(1).where(lambda s: s > 0)
        gap = (data["open"] / prior_close - 1.0).replace([np.inf, -np.inf], np.nan)
        out["gap"] = gap.fillna(0.0)
        up = gap.gt(params.min_gap)
        down = gap.lt(-params.min_gap)
        out["up_streak"] = up.astype(int).groupby((~up).cumsum()).cumsum().clip(upper=params.streak_bars)
        out["down_streak"] = down.astype(int).groupby((~down).cumsum()).cumsum().clip(upper=params.streak_bars)

        span = data["high"] - data["low"]
        location = ((2.0 * data["close"] - data["high"] - data["low"]) / span.where(span > 0)).replace([np.inf, -np.inf], np.nan)
        location = location.mask(span.eq(0), 0.0).clip(-1.0, 1.0)
        volume = data["volume"].where(data["volume"] >= 0).replace([np.inf, -np.inf], np.nan)
        weighted_pressure = location * volume
        pressure_sum = weighted_pressure.rolling(params.pressure_bars, min_periods=params.pressure_bars).sum()
        volume_sum = volume.rolling(params.pressure_bars, min_periods=params.pressure_bars).sum()
        pressure = (pressure_sum / volume_sum.where(volume_sum > 0)).replace([np.inf, -np.inf], np.nan)
        out["close_location"] = location.fillna(0.0)
        out["weighted_pressure"] = weighted_pressure.fillna(0.0)
        out["pressure_sum"] = pressure_sum.fillna(0.0)
        out["volume_sum"] = volume_sum.fillna(0.0)
        out["pressure"] = pressure.fillna(0.0)
        out["ready"] = pressure.notna() & gap.notna()
        return out

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: GapPressureParams) -> SignalFrame:
        long_entry = (
            indicators["ready"]
            & indicators["down_streak"].ge(params.streak_bars)
            & indicators["pressure"].gt(params.pressure_threshold)
        )
        short_entry = (
            indicators["ready"]
            & indicators["up_streak"].ge(params.streak_bars)
            & indicators["pressure"].lt(-params.pressure_threshold)
        )
        entries = np.select([long_entry, short_entry], [1, -1], default=0).astype(int)
        entries[:self.warmup_bars(params) - 1] = 0
        signal = np.zeros(len(data), dtype=int)
        position = 0
        exit_bar = -1
        for i in range(len(data)):
            if position != 0:
                if i == exit_bar:
                    position = 0
                else:
                    signal[i] = position
                continue
            if entries[i] != 0:
                position = int(entries[i])
                exit_bar = i + params.hold_bars
                signal[i] = position

        df = pd.DataFrame(index=data.index)
        df["signal"] = signal
        df["size"] = 0.95
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")
