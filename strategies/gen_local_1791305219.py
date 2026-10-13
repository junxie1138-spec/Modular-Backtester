from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class ParticipationParams:
    trend_lookback: int = 20
    volume_lookback: int = 40
    volume_multiple: float = 1.15
    minimum_efficiency: float = 0.35
    participation_advantage: float = 0.25
    holding_bars: int = 18


class GeneratedStrategy(BaseStrategy[ParticipationParams]):
    strategy_id = "gen_local_1791305219"

    @classmethod
    def params_type(cls) -> type[ParticipationParams]:
        return ParticipationParams

    def warmup_bars(self, params: ParticipationParams) -> int:
        return max(params.trend_lookback + 1, params.volume_lookback + 1) + 1

    def indicators(self, data: pd.DataFrame, params: ParticipationParams) -> pd.DataFrame:
        if params.trend_lookback < 2 or params.volume_lookback < 2 or params.holding_bars < 1:
            raise ValueError("Lookbacks must be at least two and holding_bars must be positive")
        if params.volume_multiple <= 0:
            raise ValueError("volume_multiple must be positive")
        out = pd.DataFrame(index=data.index)
        out["return"] = data["close"].pct_change(fill_method=None).replace([np.inf, -np.inf], np.nan)
        out["baseline_volume"] = data["volume"].shift(1).rolling(params.volume_lookback, min_periods=params.volume_lookback).mean()
        out["relative_volume"] = data["volume"].div(out["baseline_volume"].where(out["baseline_volume"] > 0))
        out["valid"] = out["return"].notna() & out["relative_volume"].notna() & data["volume"].ge(0)
        out["active"] = out["valid"] & out["relative_volume"].ge(params.volume_multiple)
        out["active_return"] = out["return"].where(out["active"], 0.0).where(out["valid"])
        out["quiet_return"] = out["return"].where(~out["active"], 0.0).where(out["valid"])
        window = params.trend_lookback
        out["active_net"] = out["active_return"].rolling(window, min_periods=window).sum()
        out["active_path"] = out["active_return"].abs().rolling(window, min_periods=window).sum()
        out["quiet_net"] = out["quiet_return"].rolling(window, min_periods=window).sum()
        out["quiet_path"] = out["quiet_return"].abs().rolling(window, min_periods=window).sum()
        out["active_efficiency"] = out["active_net"].div(out["active_path"].where(out["active_path"] > 0))
        out["quiet_efficiency"] = out["quiet_net"].div(out["quiet_path"].where(out["quiet_path"] > 0))
        out["participation_edge"] = out["active_efficiency"] - out["quiet_efficiency"]
        return out

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: ParticipationParams) -> SignalFrame:
        ready = indicators[["active_efficiency", "quiet_efficiency", "participation_edge"]].notna().all(axis=1)
        long_setup = (
            ready
            & indicators["active"]
            & indicators["return"].gt(0)
            & indicators["active_efficiency"].ge(params.minimum_efficiency)
            & indicators["participation_edge"].ge(params.participation_advantage)
        )
        short_setup = (
            ready
            & indicators["active"]
            & indicators["return"].lt(0)
            & indicators["active_efficiency"].le(-params.minimum_efficiency)
            & indicators["participation_edge"].le(-params.participation_advantage)
        )
        candidate = pd.Series(np.select([long_setup, short_setup], [1, -1], default=0), index=data.index, dtype=int)
        confirmed = candidate.where(candidate.ne(0) & candidate.eq(candidate.shift(1)), 0)
        signals = np.zeros(len(data), dtype=int)
        position = 0
        entry_bar = -1
        for bar, direction in enumerate(confirmed.to_numpy()):
            if position != 0:
                if bar - entry_bar == params.holding_bars:
                    position = 0
                    entry_bar = -1
            elif direction != 0:
                position = int(direction)
                entry_bar = bar
            signals[bar] = position
        df = pd.DataFrame({"signal": signals, "size": 1.0}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")
