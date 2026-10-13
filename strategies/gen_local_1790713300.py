from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class GeneratedParams:
    frac_order: float = 0.4
    frac_window: int = 30
    norm_window: int = 60
    entry_z: float = 1.5
    volume_window: int = 20
    volume_ratio: float = 0.8
    hold_bars: int = 15


class GeneratedStrategy(BaseStrategy[GeneratedParams]):
    strategy_id = "gen_local_1790713300"

    @classmethod
    def params_type(cls):
        return GeneratedParams

    @classmethod
    def warmup_bars(cls, params: GeneratedParams) -> int:
        return max(params.frac_window + params.norm_window, params.volume_window) + 1

    @classmethod
    def indicators(cls, data: pd.DataFrame, params: GeneratedParams) -> pd.DataFrame:
        weights = [1.0]
        for k in range(1, params.frac_window):
            weights.append(-weights[-1] * (params.frac_order - k + 1) / k)
        fd = sum(w * data["close"].shift(k) for k, w in enumerate(weights))
        scale = fd.rolling(params.norm_window).std().replace(0, np.nan)
        return pd.DataFrame({
            "fd_z": fd / scale,
            "volume_ratio": data["volume"] / data["volume"].rolling(params.volume_window).mean().replace(0, np.nan),
        }, index=data.index)

    @classmethod
    def generate_signals(cls, data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: GeneratedParams) -> SignalFrame:
        candidate = ((indicators["fd_z"] <= -params.entry_z) & (indicators["volume_ratio"] <= params.volume_ratio)).fillna(False).to_numpy()
        raw = np.zeros(len(data), dtype=int)
        remaining = 0
        for i, hit in enumerate(candidate):
            if remaining:
                remaining -= 1
            elif hit:
                raw[i] = 1
                remaining = max(1, params.hold_bars) - 1
        df = pd.DataFrame({"signal": raw, "size": 0.95}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")
