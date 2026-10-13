from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class GapPressureParams:
    volume_lookback: int = 20
    relative_volume: float = 1.2
    min_gap: float = 0.003
    profit_target: float = 0.006
    max_hold_bars: int = 2


class GeneratedStrategy(BaseStrategy[GapPressureParams]):
    strategy_id = "gen_local_1790982455"

    @classmethod
    def params_type(cls) -> type[GapPressureParams]:
        return GapPressureParams

    def warmup_bars(self, params: GapPressureParams) -> int:
        return max(2, int(params.volume_lookback)) + 2

    def indicators(self, data: pd.DataFrame, params: GapPressureParams) -> pd.DataFrame:
        result = pd.DataFrame(index=data.index)
        lookback = max(2, int(params.volume_lookback))
        previous_close = data["close"].shift(1).where(lambda x: x > 0)
        valid_open = data["open"].where(lambda x: x > 0)
        result["gap"] = data["open"] / previous_close - 1.0
        result["session_return"] = data["close"] / valid_open - 1.0
        result["previous_session_return"] = result["session_return"].shift(1)
        result["volume_baseline"] = data["volume"].shift(1).rolling(lookback, min_periods=lookback).mean()
        baseline = result["volume_baseline"].where(result["volume_baseline"] > 0)
        result["relative_volume"] = data["volume"] / baseline
        result["previous_relative_volume"] = result["relative_volume"].shift(1)
        result = result.replace([np.inf, -np.inf], np.nan)

        gap_direction = np.sign(result["gap"].fillna(0.0))
        session_direction = np.sign(result["session_return"].fillna(0.0))
        gap_qualifies = (
            (result["gap"].abs() >= max(0.0, params.min_gap))
            & (result["gap"].abs() > result["previous_session_return"].abs())
            & (result["gap"] * result["previous_session_return"] < 0)
            & (result["previous_relative_volume"] > 0)
            & (result["previous_relative_volume"] < 1.0)
        ).fillna(False)
        volume_qualifies = (
            result["relative_volume"] >= max(1.0, params.relative_volume)
        ).fillna(False)
        result["gap_vote"] = np.where(gap_qualifies, gap_direction, 0).astype(int)
        result["volume_vote"] = np.where(volume_qualifies, session_direction, 0).astype(int)
        result["entry_vote"] = np.where(
            (result["gap_vote"] != 0)
            & (result["gap_vote"] == result["volume_vote"]),
            result["gap_vote"],
            0,
        ).astype(int)
        return result

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: GapPressureParams,
    ) -> SignalFrame:
        votes = indicators["entry_vote"].fillna(0).to_numpy(dtype=int)
        opens = data["open"].to_numpy(dtype=float)
        closes = data["close"].to_numpy(dtype=float)
        decisions = np.zeros(len(data), dtype=int)
        position = 0
        entry_price = np.nan
        held_bars = 0
        hold_limit = min(2, max(1, int(params.max_hold_bars)))
        target = max(0.0001, float(params.profit_target))
        warmup = self.warmup_bars(params)

        for i in range(len(data)):
            if position != 0:
                # Yesterday's decision enters at today's open.
                if not np.isfinite(entry_price):
                    entry_price = opens[i]
                held_bars += 1
                valid_entry = np.isfinite(entry_price) and entry_price > 0
                gain = position * (closes[i] / entry_price - 1.0) if valid_entry else np.nan
                profit_exit = np.isfinite(gain) and gain >= target
                if not valid_entry or profit_exit or held_bars >= hold_limit:
                    position = 0
                    entry_price = np.nan
                    held_bars = 0
                decisions[i] = position
                continue

            if i >= warmup - 1:
                position = int(votes[i])
                entry_price = np.nan
                held_bars = 0
            decisions[i] = position

        df = pd.DataFrame(index=data.index)
        df["signal"] = decisions
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = 1.0
        assert df["signal"].isin([-1, 0, 1]).all()
        assert np.array_equal(df["signal"].to_numpy()[1:], decisions[:-1])
        return SignalFrame(data=df, signal_column="signal", size_column="size")
