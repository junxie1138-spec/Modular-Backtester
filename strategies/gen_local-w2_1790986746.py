from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class AuctionAbsorptionParams:
    breakout_bars: int = 20
    volume_bars: int = 20
    volume_multiple: float = 1.4
    absorption_bars: int = 10
    min_rejections: int = 3
    hold_bars: int = 8


class GeneratedStrategy(BaseStrategy[AuctionAbsorptionParams]):
    strategy_id = "gen_local-w2_1790986746"

    @classmethod
    def params_type(cls) -> type[AuctionAbsorptionParams]:
        return AuctionAbsorptionParams

    def warmup_bars(self, params: AuctionAbsorptionParams) -> int:
        return int(max(params.breakout_bars + 1, params.volume_bars + 1,
                       params.absorption_bars + 2))

    def indicators(self, data: pd.DataFrame, params: AuctionAbsorptionParams) -> pd.DataFrame:
        out = pd.DataFrame(index=data.index)
        previous_low = data["low"].shift(1)
        valid_probe = data[["open", "low", "close"]].notna().all(axis=1) & previous_low.notna()
        rejected_probe = ((data["low"] < previous_low)
                          & (data["close"] > previous_low)
                          & (data["close"] > data["open"]))
        probe = rejected_probe.astype(float).where(valid_probe)
        out["previous_low"] = previous_low
        out["rejected_probe"] = probe
        out["channel_ceiling"] = data["high"].rolling(
            params.breakout_bars, min_periods=params.breakout_bars
        ).max().shift(1)
        out["volume_baseline"] = data["volume"].rolling(
            params.volume_bars, min_periods=params.volume_bars
        ).mean().shift(1)
        out["prior_rejections"] = probe.rolling(
            params.absorption_bars, min_periods=params.absorption_bars
        ).sum().shift(1)
        return out.replace([np.inf, -np.inf], np.nan)

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame,
                         ctx: StrategyContext, params: AuctionAbsorptionParams) -> SignalFrame:
        volume_breakout = (
            (data["close"] > indicators["channel_ceiling"])
            & (data["close"] > data["open"])
            & (indicators["volume_baseline"] > 0)
            & (data["volume"] >= params.volume_multiple * indicators["volume_baseline"])
        )
        absorption = indicators["prior_rejections"] >= params.min_rejections
        ready = indicators[["channel_ceiling", "volume_baseline", "prior_rejections"]].notna().all(axis=1)
        entries = (volume_breakout & absorption & ready).fillna(False).to_numpy(dtype=bool)
        signal = np.zeros(len(data), dtype=int)
        exit_bar = -1
        holding = max(1, int(params.hold_bars))
        for i in range(len(data)):
            if exit_bar >= 0:
                if i < exit_bar:
                    signal[i] = 1
                else:
                    exit_bar = -1
                continue
            if entries[i]:
                signal[i] = 1
                exit_bar = i + holding
        df = pd.DataFrame(index=data.index)
        df["signal"] = signal
        df["size"] = 0.95
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")
