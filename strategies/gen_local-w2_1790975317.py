from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class StreakParams:
    streak_min: int = 3
    atr_multiple: float = 3.0


class GeneratedStrategy(BaseStrategy[StreakParams]):
    strategy_id = "gen_local-w2_1790975317"

    @classmethod
    def params_type(cls) -> type[StreakParams]:
        return StreakParams

    def warmup_bars(self, params: StreakParams) -> int:
        return max(65, params.streak_min + 2)

    def indicators(self, data: pd.DataFrame, params: StreakParams) -> pd.DataFrame:
        if params.streak_min < 1:
            raise ValueError("streak_min must be positive")
        if not np.isfinite(params.atr_multiple) or params.atr_multiple <= 0:
            raise ValueError("atr_multiple must be finite and positive")

        close = data["close"]
        change = close.diff()
        previous_close = close.shift(1)
        returns = change.div(previous_close.replace(0.0, np.nan))
        returns = returns.replace([np.inf, -np.inf], np.nan)
        down = change.lt(0)
        groups = (~down).cumsum()
        true_range = pd.concat(
            [
                data["high"] - data["low"],
                (data["high"] - previous_close).abs(),
                (data["low"] - previous_close).abs(),
            ],
            axis=1,
        ).max(axis=1)

        result = pd.DataFrame(index=data.index)
        result["return"] = returns
        result["down_streak"] = down.groupby(groups).cumsum().astype(int)
        result["streak_drop"] = (-change.where(down, 0.0)).groupby(groups).cumsum()
        result["atr"] = true_range.rolling(14, min_periods=14).mean()
        result["autocorrelation"] = returns.rolling(63, min_periods=63).corr(returns.shift(1))
        result["autocorrelation"] = result["autocorrelation"].replace([np.inf, -np.inf], np.nan)
        result["entry_ready"] = (
            result["down_streak"].shift(1).ge(params.streak_min)
            & result["streak_drop"].shift(1).ge(1.5 * result["atr"].shift(1))
            & returns.gt(0)
            & result["autocorrelation"].lt(0)
            & result["atr"].gt(0)
        ).fillna(False)
        return result

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: StreakParams,
    ) -> SignalFrame:
        closes = data["close"].to_numpy(dtype=float)
        atrs = indicators["atr"].to_numpy(dtype=float)
        entries = indicators["entry_ready"].fillna(False).to_numpy(dtype=bool)
        signals = np.zeros(len(data), dtype=int)
        holding = False
        entry_bar = -1
        high_water = 0.0
        stop = 0.0
        first_bar = self.warmup_bars(params) - 1

        for i in range(len(data)):
            close = closes[i]
            atr = atrs[i]
            valid = np.isfinite(close) and np.isfinite(atr) and atr > 0
            if holding:
                expired = i - entry_bar >= 18
                stopped = False
                if valid:
                    high_water = max(high_water, close)
                    stop = max(stop, high_water - params.atr_multiple * atr)
                    stopped = close <= stop
                if expired or stopped:
                    holding = False
            elif i >= first_bar and valid and entries[i]:
                holding = True
                entry_bar = i
                high_water = close
                stop = close - params.atr_multiple * atr
            signals[i] = int(holding)

        df = pd.DataFrame(index=data.index)
        df["signal"] = signals
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = 0.95
        return SignalFrame(data=df, signal_column="signal", size_column="size")
