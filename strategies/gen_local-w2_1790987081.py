from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class AuctionOverlapParams:
    expansion_threshold: float = 1.35
    overlap_threshold: float = 0.60
    holding_bars: int = 18


class GeneratedStrategy(BaseStrategy[AuctionOverlapParams]):
    strategy_id = "gen_local-w2_1790987081"

    @classmethod
    def params_type(cls) -> type[AuctionOverlapParams]:
        return AuctionOverlapParams

    def warmup_bars(self, params: AuctionOverlapParams) -> int:
        return 9

    def indicators(self, data: pd.DataFrame, params: AuctionOverlapParams) -> pd.DataFrame:
        out = pd.DataFrame(index=data.index)
        out["range"] = (data["high"] - data["low"]).where(lambda x: x > 0)
        out["previous_high"] = data["high"].shift(1)
        out["previous_low"] = data["low"].shift(1)
        out["intersection"] = (
            np.minimum(data["high"], out["previous_high"])
            - np.maximum(data["low"], out["previous_low"])
        ).clip(lower=0)
        out["smaller_range"] = np.minimum(out["range"], out["range"].shift(1))
        out["overlap"] = (out["intersection"] / out["smaller_range"]).clip(0, 1)
        out["overlap_regime"] = out["overlap"].rolling(8, min_periods=8).mean()
        out["baseline_range"] = out["range"].shift(1).rolling(8, min_periods=8).mean()
        out["expansion"] = out["range"] / out["baseline_range"]
        out["close_location"] = ((data["close"] - data["low"]) / out["range"]).clip(0, 1)
        return out.replace([np.inf, -np.inf], np.nan)

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: AuctionOverlapParams,
    ) -> SignalFrame:
        valid = indicators[["overlap_regime", "expansion", "close_location", "previous_high", "previous_low"]].notna().all(axis=1)
        expanded = indicators["expansion"] >= params.expansion_threshold
        upside = (data["close"] > indicators["previous_high"]) & (indicators["close_location"] >= 0.75)
        downside = (data["close"] < indicators["previous_low"]) & (indicators["close_location"] <= 0.25)
        separated = indicators["overlap_regime"] <= 1.0 - params.overlap_threshold
        congested = indicators["overlap_regime"] >= params.overlap_threshold
        breakout = pd.Series(np.select([upside, downside], [1, -1], default=0), index=data.index)
        regime_direction = pd.Series(np.select([separated, congested], [1, -1], default=0), index=data.index)
        candidates = (breakout * regime_direction).where(valid & expanded, 0).astype(int).to_numpy()

        signals = np.zeros(len(data), dtype=int)
        position = 0
        exit_bar = -1
        holding_bars = max(1, int(params.holding_bars))
        for i in range(len(data)):
            if position != 0:
                if i == exit_bar:
                    position = 0
                signals[i] = position
                continue
            if i >= self.warmup_bars(params) - 1 and candidates[i] != 0:
                position = int(candidates[i])
                exit_bar = i + holding_bars
            signals[i] = position

        df = pd.DataFrame(index=data.index)
        df["signal"] = signals
        df["size"] = 1.0
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")
