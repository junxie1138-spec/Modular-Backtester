from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class ParticipationParams:
    volume_window: int = 20
    backlog_window: int = 5
    volume_multiple: float = 1.6
    backlog_threshold: float = 1.25
    impulse_threshold: float = 0.75
    close_location_threshold: float = 0.75


class GeneratedStrategy(BaseStrategy[ParticipationParams]):
    strategy_id = "gen_local_1790985721"

    @classmethod
    def params_type(cls) -> type[ParticipationParams]:
        return ParticipationParams

    @staticmethod
    def warmup_bars(params: ParticipationParams) -> int:
        return int(max(200, params.volume_window + params.backlog_window + 1))

    @staticmethod
    def indicators(data: pd.DataFrame, params: ParticipationParams) -> pd.DataFrame:
        out = pd.DataFrame(index=data.index)
        out["ma200"] = data["close"].rolling(200, min_periods=200).mean()
        out["body_return"] = data["close"].div(data["open"].replace(0.0, np.nan)).sub(1.0)
        out["volume_baseline"] = data["volume"].rolling(
            params.volume_window, min_periods=params.volume_window
        ).mean().shift(1)
        out["body_scale"] = out["body_return"].abs().rolling(
            params.volume_window, min_periods=params.volume_window
        ).mean().shift(1)
        out["relative_volume"] = data["volume"].div(
            out["volume_baseline"].replace(0.0, np.nan)
        )
        out["normalized_body"] = out["body_return"].div(
            out["body_scale"].replace(0.0, np.nan)
        )
        out["participation_gap"] = (1.0 - out["relative_volume"]).clip(0.0, 1.0)
        out["signed_debt"] = out["normalized_body"] * out["participation_gap"]
        out["backlog"] = out["signed_debt"].shift(1).rolling(
            params.backlog_window, min_periods=params.backlog_window
        ).sum()
        out["close_location"] = (data["close"] - data["low"]).div(
            (data["high"] - data["low"]).replace(0.0, np.nan)
        )
        return out.replace([np.inf, -np.inf], np.nan)

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: ParticipationParams,
    ) -> SignalFrame:
        ready = indicators.notna().all(axis=1)
        surge = indicators["relative_volume"] >= params.volume_multiple
        long_entry = (
            ready
            & surge
            & (data["close"] > indicators["ma200"])
            & (indicators["backlog"] >= params.backlog_threshold)
            & (indicators["normalized_body"] >= params.impulse_threshold)
            & (indicators["close_location"] >= params.close_location_threshold)
        )
        short_entry = (
            ready
            & surge
            & (data["close"] < indicators["ma200"])
            & (indicators["backlog"] <= -params.backlog_threshold)
            & (indicators["normalized_body"] <= -params.impulse_threshold)
            & (indicators["close_location"] <= 1.0 - params.close_location_threshold)
        )
        candidates = np.select([long_entry, short_entry], [1, -1], default=0)
        signals = np.zeros(len(data), dtype=int)
        position = 0
        entry_bar = -1
        for i, direction in enumerate(candidates):
            if position:
                if i - entry_bar == 2:
                    position = 0
            elif direction:
                position = int(direction)
                entry_bar = i
            signals[i] = position
        df = pd.DataFrame({"signal": signals, "size": 1.0}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")
