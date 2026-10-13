from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class DrawdownImpulseParams:
    drawdown_window: int = 12
    impulse_bars: int = 3
    reversal_threshold: float = 0.3
    target_daily_vol: float = 0.01


class GeneratedStrategy(BaseStrategy[DrawdownImpulseParams]):
    strategy_id = "gen_local_1791297170"

    @classmethod
    def params_type(cls) -> type[DrawdownImpulseParams]:
        return DrawdownImpulseParams

    def warmup_bars(self, params: DrawdownImpulseParams) -> int:
        return max(200, params.drawdown_window + params.impulse_bars + 2, 21)

    def indicators(self, data: pd.DataFrame, params: DrawdownImpulseParams) -> pd.DataFrame:
        close = data["close"]
        ma200 = close.rolling(200, min_periods=200).mean()
        peak = close.rolling(params.drawdown_window, min_periods=params.drawdown_window).max()
        depth = (1.0 - close / peak).clip(lower=0).fillna(0.0)
        depth_impulse = depth.rolling(3, min_periods=3).mean().diff(params.impulse_bars).fillna(0.0)
        daily_vol = close.pct_change().rolling(20, min_periods=20).std().fillna(0.0)
        size = (params.target_daily_vol / daily_vol.clip(lower=0.002)).clip(lower=0.1, upper=0.95)
        return pd.DataFrame({
            "ma200": ma200.fillna(0.0),
            "ready": ma200.notna(),
            "depth_impulse": depth_impulse,
            "daily_vol": daily_vol,
            "size": size,
        }, index=data.index)

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: DrawdownImpulseParams,
    ) -> SignalFrame:
        threshold = params.reversal_threshold * indicators["daily_vol"]
        recovering = indicators["ready"] & (indicators["depth_impulse"] < -threshold)
        deepening = indicators["ready"] & (indicators["depth_impulse"] > threshold)
        long_entry = recovering & (data["close"] > indicators["ma200"])
        short_entry = deepening & (data["close"] < indicators["ma200"])

        position = 0
        signals = []
        for enter_long, enter_short, up, down in zip(long_entry, short_entry, recovering, deepening):
            if position == 1 and down:
                position = 0
            elif position == -1 and up:
                position = 0
            elif position == 0:
                if enter_long:
                    position = 1
                elif enter_short:
                    position = -1
            signals.append(position)

        df = pd.DataFrame(index=data.index)
        df["signal"] = pd.Series(signals, index=data.index, dtype=int)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = indicators["size"].astype(float)
        return SignalFrame(data=df, signal_column="signal", size_column="size")
