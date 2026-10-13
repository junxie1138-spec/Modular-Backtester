from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class WaveletParams:
    coarse_threshold: float = 0.65
    min_impulse: float = 0.35
    hold_bars: int = 8


class GeneratedStrategy(BaseStrategy[WaveletParams]):
    strategy_id = "gen_local-w2_1791300224"

    @classmethod
    def params_type(cls):
        return WaveletParams

    def warmup_bars(self, params: WaveletParams) -> int:
        return 21

    def indicators(self, data: pd.DataFrame, params: WaveletParams) -> pd.DataFrame:
        log_close = np.log(data["close"].clip(lower=1e-12))
        d2 = log_close.rolling(2).mean() - log_close.rolling(2).mean().shift(2)
        d4 = log_close.rolling(4).mean() - log_close.rolling(4).mean().shift(4)
        d8 = log_close.rolling(8).mean() - log_close.rolling(8).mean().shift(8)
        energy = d2.pow(2) + d4.pow(2) + d8.pow(2)
        coarse_share = (d4.pow(2) + d8.pow(2)) / (energy + 1e-12)
        volatility = log_close.diff().rolling(20).std()
        impulse = d8 / (volatility * np.sqrt(8) + 1e-12)
        out = pd.DataFrame({
            "d2": d2,
            "d4": d4,
            "d8": d8,
            "coarse_share": coarse_share,
            "impulse": impulse,
        }, index=data.index)
        return out.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: WaveletParams,
    ) -> SignalFrame:
        share = indicators["coarse_share"]
        impulse = indicators["impulse"]
        entry = (
            share.gt(params.coarse_threshold)
            & share.shift(1).le(params.coarse_threshold)
            & indicators["d4"].gt(0)
            & indicators["d8"].gt(0)
            & impulse.gt(params.min_impulse)
            & (np.arange(len(data)) >= self.warmup_bars(params))
        ).fillna(False)

        raw = np.zeros(len(data), dtype=int)
        exit_at = -1
        for i, trigger in enumerate(entry.to_numpy()):
            if i == exit_at:
                continue
            if i < exit_at:
                raw[i] = 1
            elif trigger:
                raw[i] = 1
                exit_at = i + params.hold_bars

        conviction = ((impulse - params.min_impulse) / 1.5).clip(0.0, 1.0)
        df = pd.DataFrame(index=data.index)
        df["signal"] = raw
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = (0.25 + 0.75 * conviction).shift(1).fillna(0.25).astype(float)
        return SignalFrame(data=df, signal_column="signal", size_column="size")
