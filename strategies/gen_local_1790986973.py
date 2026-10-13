from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class StreakAlternationParams:
    streak_length: int = 2
    correlation_window: int = 20
    negative_correlation_threshold: float = 0.10


class GeneratedStrategy(BaseStrategy[StreakAlternationParams]):
    strategy_id = "gen_local_1790986973"

    @classmethod
    def params_type(cls) -> type[StreakAlternationParams]:
        return StreakAlternationParams

    @staticmethod
    def warmup_bars(params: StreakAlternationParams) -> int:
        return int(max(params.correlation_window + 1, params.streak_length + 1))

    @staticmethod
    def indicators(data: pd.DataFrame, params: StreakAlternationParams) -> pd.DataFrame:
        out = pd.DataFrame(index=data.index)
        change = data["close"].diff()
        direction = np.sign(change).fillna(0).astype(int)
        groups = direction.ne(direction.shift(1)).cumsum()
        streak = direction.groupby(groups).cumcount().add(1)
        streak = streak.where(direction.ne(0), 0)
        intraday = data["close"].div(data["open"].replace(0, np.nan)).sub(1)
        intraday = intraday.replace([np.inf, -np.inf], np.nan)
        correlation = intraday.rolling(
            params.correlation_window, min_periods=params.correlation_window
        ).corr(intraday.shift(1))
        correlation = correlation.replace([np.inf, -np.inf], np.nan)
        out["close_change"] = change.fillna(0.0)
        out["close_direction"] = direction
        out["streak_count"] = streak.astype(int)
        out["intraday_return"] = intraday.fillna(0.0)
        out["intraday_direction"] = np.sign(intraday).fillna(0).astype(int)
        out["intraday_autocorrelation"] = correlation.fillna(0.0)
        out["ready"] = correlation.notna() & intraday.notna() & change.notna()
        out["streak_forecast"] = (-direction).where(
            streak.ge(params.streak_length), 0
        ).astype(int)
        out["intraday_forecast"] = (-out["intraday_direction"]).where(
            out["ready"] & correlation.lt(-params.negative_correlation_threshold), 0
        ).astype(int)
        return out

    @staticmethod
    def generate_signals(
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: StreakAlternationParams,
    ) -> SignalFrame:
        agree = (
            indicators["ready"]
            & indicators["streak_forecast"].ne(0)
            & indicators["streak_forecast"].eq(indicators["intraday_forecast"])
        )
        entries = indicators["streak_forecast"].where(agree, 0).to_numpy(dtype=int)
        signals = np.zeros(len(data), dtype=int)
        position = 0
        for i, entry in enumerate(entries):
            if position == 0:
                position = int(entry)
            elif entry == -position:
                # Opposite joint forecast closes the position; this bar stays flat.
                position = 0
            signals[i] = position
        df = pd.DataFrame(index=data.index)
        df["signal"] = signals
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = 0.95
        return SignalFrame(data=df, signal_column="signal", size_column="size")
