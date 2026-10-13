from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class ExpiryStreakParams:
    min_streak: int = 3
    full_size_streak: int = 6
    hold_bars: int = 4
    min_size: float = 0.35
    max_size: float = 0.95


class GeneratedStrategy(BaseStrategy[ExpiryStreakParams]):
    strategy_id = "gen_local-w2_1791249109"

    @classmethod
    def params_type(cls) -> type[ExpiryStreakParams]:
        return ExpiryStreakParams

    def warmup_bars(self, params: ExpiryStreakParams) -> int:
        return int(max(params.min_streak, params.full_size_streak) + 1)

    def indicators(self, data: pd.DataFrame, params: ExpiryStreakParams) -> pd.DataFrame:
        if not (1 <= params.min_streak <= params.full_size_streak):
            raise ValueError("Require 1 <= min_streak <= full_size_streak.")
        if not (3 <= params.hold_bars <= 5):
            raise ValueError("hold_bars must be between 3 and 5.")
        if not (0 < params.min_size <= params.max_size <= 1):
            raise ValueError("Require 0 < min_size <= max_size <= 1.")
        close = data["close"]
        valid = np.isfinite(close) & np.isfinite(close.shift(1)) & (close > 0) & (close.shift(1) > 0)
        direction = np.sign(close.diff()).where(valid, 0).fillna(0).astype(int)
        groups = direction.ne(direction.shift(1)).cumsum()
        streak = (direction.groupby(groups).cumcount() + 1).where(direction.ne(0), 0)
        streak = streak.clip(upper=params.full_size_streak).astype(int)
        weekday = data.index.dayofweek
        upcoming_friday_day = data.index.day + ((4 - weekday) % 7)
        # Calendar anchor only; no exchange holiday calendar is required.
        expiry_window = (weekday >= 1) & (weekday <= 3) & (upcoming_friday_day >= 15) & (upcoming_friday_day <= 21)
        eligible = expiry_window & streak.ge(params.min_streak) & direction.ne(0)
        entry = (-direction).where(eligible, 0).astype(int)
        strength = (streak / params.full_size_streak).clip(0, 1)
        entry_size = params.min_size + (params.max_size - params.min_size) * strength
        return pd.DataFrame(
            {"direction": direction, "streak": streak, "expiry_window": expiry_window,
             "entry": entry, "strength": strength, "entry_size": entry_size},
            index=data.index,
        )

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame,
                         ctx: StrategyContext, params: ExpiryStreakParams) -> SignalFrame:
        signals = np.zeros(len(data), dtype=int)
        sizes = np.full(len(data), params.min_size, dtype=float)
        entries = indicators["entry"].fillna(0).to_numpy(dtype=int)
        entry_sizes = indicators["entry_size"].fillna(params.min_size).to_numpy(dtype=float)
        position = 0
        exit_bar = -1
        position_size = params.min_size
        for i in range(len(data)):
            if position:
                if i == exit_bar:
                    position = 0
                    continue
            elif entries[i]:
                position = int(entries[i])
                position_size = float(entry_sizes[i])
                exit_bar = i + params.hold_bars
            signals[i] = position
            if position:
                sizes[i] = position_size
        df = pd.DataFrame({"signal": signals, "size": sizes}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = df["size"].shift(1).fillna(params.min_size).astype(float)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check() -> None:
    index = pd.bdate_range("2024-01-08", periods=16)
    strategy = GeneratedStrategy()
    for direction in (-1, 1):
        close = 100.0 + direction * np.arange(len(index))
        data = pd.DataFrame(
            {"open": close, "high": close + 1, "low": close - 1,
             "close": close, "volume": 1000000.0}, index=index,
        )
        for hold in (3, 4, 5):
            params = ExpiryStreakParams(hold_bars=hold)
            features = strategy.indicators(data, params)
            result = strategy.generate_signals(data, features, None, params).data
            active = np.flatnonzero(result["signal"].to_numpy())
            assert np.array_equal(active, np.arange(7, 7 + hold))
            assert (result["signal"].iloc[active] == -direction).all()
            assert result["size"].iloc[active].nunique() == 1
            assert (result["size"] > 0).all()
            prefix = data.iloc[:9]
            partial = strategy.generate_signals(
                prefix, strategy.indicators(prefix, params), None, params,
            ).data
            assert partial.equals(result.iloc[:9])


if __name__ == "__main__":
    _self_check()
