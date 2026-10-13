from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class StreakParams:
    percentile_window: int = 63
    maturity_percentile: float = 0.70
    streak_cap: int = 6


class GeneratedStrategy(BaseStrategy[StreakParams]):
    strategy_id = "gen_local_1791272659"

    @classmethod
    def params_type(cls) -> type[StreakParams]:
        return StreakParams

    def warmup_bars(self, params: StreakParams) -> int:
        return int(params.percentile_window + params.streak_cap + 1)

    def indicators(self, data: pd.DataFrame, params: StreakParams) -> pd.DataFrame:
        if params.percentile_window < 2 or params.streak_cap < 2:
            raise ValueError("Lookback and streak cap must be at least two")
        if not 0.0 < params.maturity_percentile < 1.0:
            raise ValueError("Maturity percentile must be between zero and one")
        direction = np.sign(data["close"].diff()).fillna(0).astype(int)
        active = direction.ne(0)
        streak = active.astype(int)
        for lag in range(1, params.streak_cap):
            active = active & direction.eq(direction.shift(lag))
            streak = streak + active.astype(int)
        threshold = streak.shift(1).rolling(
            params.percentile_window,
            min_periods=params.percentile_window,
        ).quantile(params.maturity_percentile)
        forecast = direction.where(streak.lt(threshold), -direction)
        forecast = forecast.where(threshold.notna() & direction.ne(0), 0).astype(int)
        return pd.DataFrame({
            "direction": direction,
            "streak": streak,
            "maturity_threshold": threshold,
            "forecast": forecast,
        }, index=data.index)

    @staticmethod
    def _positions(forecasts: np.ndarray) -> np.ndarray:
        positions = np.zeros(len(forecasts), dtype=int)
        position = 0
        for i, forecast in enumerate(forecasts):
            candidate = int(forecast)
            if position == 0:
                position = candidate
            elif candidate == -position:
                position = 0
            positions[i] = position
        return positions

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: StreakParams,
    ) -> SignalFrame:
        forecasts = indicators["forecast"].fillna(0).astype(int).to_numpy()
        df = pd.DataFrame(index=data.index)
        df["signal"] = self._positions(forecasts)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = 1.0
        return SignalFrame(data=df, signal_column="signal", size_column="size")


if __name__ == "__main__":
    actual = GeneratedStrategy._positions(np.array([1, 1, 0, -1, -1, 1, 1]))
    assert np.array_equal(actual, np.array([1, 1, 1, 0, -1, 0, 1]))
