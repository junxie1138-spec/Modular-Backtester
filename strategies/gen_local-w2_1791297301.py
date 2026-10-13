from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class MomentumParams:
    roc_bars: int = 5
    accel_lag: int = 3
    hold_bars: int = 8
    threshold: float = 0.8


class GeneratedStrategy(BaseStrategy[MomentumParams]):
    strategy_id = "gen_local-w2_1791297301"

    @classmethod
    def params_type(cls) -> type[MomentumParams]:
        return MomentumParams

    @classmethod
    def warmup_bars(cls, params: MomentumParams) -> int:
        return max(200, 21, params.roc_bars + params.accel_lag + 1)

    def indicators(self, data: pd.DataFrame, params: MomentumParams) -> pd.DataFrame:
        close = data["close"]
        returns = close.pct_change().replace([np.inf, -np.inf], np.nan)
        roc = close.pct_change(params.roc_bars).replace([np.inf, -np.inf], np.nan)
        acceleration = roc.diff(params.accel_lag)
        volatility = returns.rolling(20, min_periods=20).std()
        ma200 = close.rolling(200, min_periods=200).mean()
        avg_volume = data["volume"].rolling(20, min_periods=20).mean()
        ready = ma200.notna() & volatility.gt(0) & avg_volume.gt(0)
        score = (acceleration / (volatility * np.sqrt(params.roc_bars))).replace([np.inf, -np.inf], np.nan)
        return pd.DataFrame({
            "score": score.fillna(0.0),
            "ma200": ma200.fillna(0.0),
            "quiet": (data["volume"] < avg_volume).fillna(False),
            "ready": ready.fillna(False),
        }, index=data.index)

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: MomentumParams) -> SignalFrame:
        long_entry = (indicators["ready"] & indicators["quiet"] &
                      data["close"].gt(indicators["ma200"]) &
                      indicators["score"].gt(params.threshold)).to_numpy()
        short_entry = (indicators["ready"] & indicators["quiet"] &
                       data["close"].lt(indicators["ma200"]) &
                       indicators["score"].lt(-params.threshold)).to_numpy()
        signal = np.zeros(len(data), dtype=int)
        position = 0
        held = 0
        for i in range(len(data)):
            if position:
                held += 1
                if held >= params.hold_bars:
                    position = 0
            elif long_entry[i] or short_entry[i]:
                position = 1 if long_entry[i] else -1
                held = 0
            signal[i] = position
        df = pd.DataFrame({"signal": signal, "size": 1.0}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")
