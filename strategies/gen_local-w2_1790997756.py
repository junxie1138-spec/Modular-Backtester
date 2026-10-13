from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class StreakParams:
    streak_length: int = 3
    autocorr_window: int = 60
    autocorr_threshold: float = -0.10
    holding_bars: int = 8


class GeneratedStrategy(BaseStrategy[StreakParams]):
    strategy_id = "gen_local-w2_1790997756"

    @classmethod
    def params_type(cls) -> type[StreakParams]:
        return StreakParams

    def warmup_bars(self, params: StreakParams) -> int:
        return int(max(params.autocorr_window + 2, params.streak_length + 1))

    def indicators(self, data: pd.DataFrame, params: StreakParams) -> pd.DataFrame:
        close = data["close"].astype(float)
        changes = close.diff()
        signs = np.sign(changes)
        down = changes.lt(0).fillna(False)
        groups = (~down).cumsum()
        down_streak = down.astype(int).groupby(groups).cumsum()
        lagged_signs = signs.shift(1)
        sign_autocorr = signs.rolling(
            window=params.autocorr_window,
            min_periods=params.autocorr_window,
        ).corr(lagged_signs)
        sign_autocorr = sign_autocorr.replace([np.inf, -np.inf], np.nan)
        result = pd.DataFrame(index=data.index)
        result["close_change"] = changes
        result["return_sign"] = signs
        result["lagged_return_sign"] = lagged_signs
        result["down_streak"] = down_streak
        result["sign_autocorr"] = sign_autocorr
        result["streak_agrees"] = down_streak.ge(params.streak_length)
        result["autocorr_agrees"] = sign_autocorr.le(params.autocorr_threshold).fillna(False)
        return result

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: StreakParams,
    ) -> SignalFrame:
        candidates = (
            indicators["streak_agrees"].fillna(False)
            & indicators["autocorr_agrees"].fillna(False)
            & indicators["sign_autocorr"].notna()
        ).to_numpy(dtype=bool)
        signal = np.zeros(len(data), dtype=int)
        holding_bars = max(1, int(params.holding_bars))
        exit_bar = -1
        active = False
        warmup = self.warmup_bars(params)
        for i in range(len(data)):
            if active:
                if i == exit_bar:
                    active = False
                    continue
                signal[i] = 1
            elif i >= warmup - 1 and candidates[i]:
                active = True
                exit_bar = i + holding_bars
                signal[i] = 1
        df = pd.DataFrame(index=data.index)
        df["signal"] = signal
        df["size"] = 1.0
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")
