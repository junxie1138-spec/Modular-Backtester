from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class GeneratedParams:
    peak_window: int = 30
    atr_window: int = 14
    long_depth: float = 3.0
    short_impulse: float = 0.8
    target_daily_vol: float = 0.012
    stop_atr: float = 1.6
    hold_bars: int = 4


class GeneratedStrategy(BaseStrategy[GeneratedParams]):
    strategy_id = "gen_local-w2_1791226884"

    @classmethod
    def params_type(cls) -> type[GeneratedParams]:
        return GeneratedParams

    @classmethod
    def warmup_bars(cls, params: GeneratedParams) -> int:
        return max(params.peak_window + 2, params.atr_window + 1)

    def indicators(self, data: pd.DataFrame, params: GeneratedParams) -> pd.DataFrame:
        previous_close = data["close"].shift(1)
        true_range = pd.concat(
            [
                data["high"] - data["low"],
                (data["high"] - previous_close).abs(),
                (data["low"] - previous_close).abs(),
            ],
            axis=1,
        ).max(axis=1)
        atr = true_range.rolling(params.atr_window, min_periods=params.atr_window).mean()
        peak = data["close"].rolling(params.peak_window, min_periods=params.peak_window).max()
        depth = ((peak - data["close"]) / peak.replace(0, np.nan)).clip(lower=0)
        daily_vol = (atr / data["close"].replace(0, np.nan)).replace(0, np.nan)
        location = ((data["close"] - data["low"]) / (data["high"] - data["low"]).replace(0, np.nan)).clip(0, 1)
        return pd.DataFrame(
            {
                "atr": atr.fillna(0),
                "daily_vol": daily_vol.fillna(0),
                "depth_units": (depth / daily_vol).fillna(0),
                "depth_impulse": (depth.diff(2) / daily_vol).fillna(0),
                "location": location.fillna(0.5),
            },
            index=data.index,
        )

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: GeneratedParams,
    ) -> SignalFrame:
        signals: list[int] = []
        sizes: list[float] = []
        position = 0
        planned = 0
        planned_size = 0.05
        active_size = 0.05
        entry_price = 0.0
        entry_atr = 0.0
        entry_bar = 0
        warmup = self.warmup_bars(params)

        for i in range(len(data)):
            if planned != position:
                position = planned
                if position:
                    entry_price = float(data["open"].iloc[i])
                    entry_atr = float(indicators["atr"].iloc[i - 1])
                    entry_bar = i
                    active_size = planned_size

            next_position = position
            next_size = active_size
            if position:
                close = float(data["close"].iloc[i])
                stopped = (
                    close < entry_price - params.stop_atr * entry_atr
                    if position == 1
                    else close > entry_price + params.stop_atr * entry_atr
                )
                if stopped or i - entry_bar + 1 >= params.hold_bars:
                    next_position = 0
            elif i >= warmup:
                depth = float(indicators["depth_units"].iloc[i])
                impulse = float(indicators["depth_impulse"].iloc[i])
                location = float(indicators["location"].iloc[i])
                if depth >= params.long_depth and impulse <= 0.15 and location >= 0.65:
                    next_position = 1
                    strength = np.clip((depth - params.long_depth) / max(params.long_depth, 1e-6), 0, 1)
                elif depth >= 1.5 and impulse >= params.short_impulse and location <= 0.35:
                    next_position = -1
                    strength = np.clip((impulse - params.short_impulse) / max(params.short_impulse, 1e-6), 0, 1)
                if next_position:
                    vol = max(float(indicators["daily_vol"].iloc[i]), 1e-6)
                    next_size = float(np.clip(params.target_daily_vol / vol * (0.25 + 0.75 * strength), 0.05, 0.95))

            planned = next_position
            planned_size = next_size
            signals.append(planned)
            sizes.append(next_size)

        df = pd.DataFrame(index=data.index)
        df["signal"] = pd.Series(signals, index=data.index)
        df["size"] = pd.Series(sizes, index=data.index).shift(1).fillna(0.05).clip(lower=0.05)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")
