from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class BreakoutParams:
    channel_bars: int = 20
    volume_bars: int = 20
    drought_ratio: float = 0.85
    surge_ratio: float = 1.5
    confirmation_ratio: float = 1.0
    hold_bars: int = 8


class GeneratedStrategy(BaseStrategy[BreakoutParams]):
    strategy_id = "gen_local-w2_1790995814"

    @classmethod
    def params_type(cls) -> type[BreakoutParams]:
        return BreakoutParams

    def warmup_bars(self, params: BreakoutParams) -> int:
        return int(max(params.channel_bars + 1, params.volume_bars + 4, 4))

    def indicators(self, data: pd.DataFrame, params: BreakoutParams) -> pd.DataFrame:
        out = pd.DataFrame(index=data.index)
        out["upper"] = data["high"].rolling(params.channel_bars, min_periods=params.channel_bars).max().shift(1)
        out["lower"] = data["low"].rolling(params.channel_bars, min_periods=params.channel_bars).min().shift(1)
        # The baseline excludes the breakout and the preceding three drought bars.
        out["volume_baseline"] = data["volume"].rolling(params.volume_bars, min_periods=params.volume_bars).mean().shift(4)
        out["pre_volume"] = data["volume"].rolling(3, min_periods=3).mean().shift(1)
        baseline = out["volume_baseline"].where(out["volume_baseline"] > 0)
        out["drought"] = out["pre_volume"] / baseline
        out["participation"] = data["volume"] / baseline
        out["confirmation_participation"] = data["volume"] / baseline.shift(1)
        out["frozen_upper"] = out["upper"].shift(1)
        out["frozen_lower"] = out["lower"].shift(1)
        out["previous_close"] = data["close"].shift(1)
        ready = out[["upper", "lower", "drought", "participation"]].notna().all(axis=1)
        drought = out["drought"] <= params.drought_ratio
        surge = out["participation"] >= params.surge_ratio
        out["long_setup"] = (ready & drought & surge & (data["close"] > out["upper"]) & (data["close"] > data["open"])).astype(int)
        out["short_setup"] = (ready & drought & surge & (data["close"] < out["lower"]) & (data["close"] < data["open"])).astype(int)
        return out.replace([np.inf, -np.inf], np.nan)

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: BreakoutParams) -> SignalFrame:
        # Both sessions must show directional closes and sufficient volume.
        participation = indicators["confirmation_participation"] >= params.confirmation_ratio
        long_entry = (
            indicators["long_setup"].shift(1).eq(1)
            & participation
            & (data["close"] > indicators["frozen_upper"])
            & (data["close"] > indicators["previous_close"])
            & (data["close"] > data["open"])
        ).fillna(False)
        short_entry = (
            indicators["short_setup"].shift(1).eq(1)
            & participation
            & (data["close"] < indicators["frozen_lower"])
            & (data["close"] < indicators["previous_close"])
            & (data["close"] < data["open"])
        ).fillna(False)
        candidates = np.where(long_entry.to_numpy(), 1, np.where(short_entry.to_numpy(), -1, 0))
        targets = np.zeros(len(data), dtype=int)
        position = 0
        exit_bar = -1
        hold_bars = max(1, int(params.hold_bars))
        for i in range(len(data)):
            if position != 0 and i == exit_bar:
                # Force a flat target; no re-entry on the expiry decision bar.
                position = 0
                continue
            if position == 0 and candidates[i] != 0:
                position = int(candidates[i])
                exit_bar = i + hold_bars
            targets[i] = position
        df = pd.DataFrame(index=data.index)
        df["signal"] = targets
        # Shifting entry and expiry together preserves exactly hold_bars bars.
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = 0.95
        return SignalFrame(data=df, signal_column="signal", size_column="size")
