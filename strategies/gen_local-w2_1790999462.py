from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class StreakParams:
    correlation_window: int = 20
    percentile_window: int = 126
    streak_quantile: float = 0.80
    correlation_quantile: float = 0.65
    atr_window: int = 14
    stop_atr: float = 1.0
    holding_bars: int = 2


class GeneratedStrategy(BaseStrategy[StreakParams]):
    strategy_id = "gen_local-w2_1790999462"

    @classmethod
    def params_type(cls):
        return StreakParams

    def warmup_bars(self, params: StreakParams) -> int:
        return int(max(params.correlation_window + params.percentile_window + 2,
                       params.atr_window + 1))

    def indicators(self, data: pd.DataFrame, params: StreakParams) -> pd.DataFrame:
        close = data["close"]
        returns = close.pct_change(fill_method=None).replace([np.inf, -np.inf], np.nan)
        rising = returns.gt(0) & returns.notna()
        groups = (~rising).cumsum()
        streak = rising.astype(int).groupby(groups).cumsum().astype(float)
        correlation = returns.rolling(
            params.correlation_window, min_periods=params.correlation_window
        ).corr(returns.shift(1)).replace([np.inf, -np.inf], np.nan)
        previous_close = close.shift(1)
        true_range = pd.concat([
            data["high"] - data["low"],
            (data["high"] - previous_close).abs(),
            (data["low"] - previous_close).abs(),
        ], axis=1).max(axis=1)
        out = pd.DataFrame(index=data.index)
        out["returns"] = returns
        out["up_streak"] = streak
        out["streak_threshold"] = streak.shift(1).rolling(
            params.percentile_window, min_periods=params.percentile_window
        ).quantile(params.streak_quantile)
        out["correlation"] = correlation
        out["correlation_threshold"] = correlation.shift(1).rolling(
            params.percentile_window, min_periods=params.percentile_window
        ).quantile(params.correlation_quantile)
        out["true_range"] = true_range
        out["atr"] = true_range.rolling(
            params.atr_window, min_periods=params.atr_window
        ).mean()
        return out

    def generate_signals(
        self, data: pd.DataFrame, indicators: pd.DataFrame,
        ctx: StrategyContext, params: StreakParams,
    ) -> SignalFrame:
        entry = (
            indicators["up_streak"].gt(indicators["streak_threshold"])
            & indicators["up_streak"].gt(0)
            & indicators["correlation"].gt(indicators["correlation_threshold"])
            & indicators["correlation"].gt(0)
            & indicators["atr"].gt(0)
        ).fillna(False).to_numpy(dtype=bool)
        opens = data["open"].to_numpy(dtype=float)
        closes = data["close"].to_numpy(dtype=float)
        atr = indicators["atr"].to_numpy(dtype=float)
        decisions = np.zeros(len(data), dtype=int)
        active = False
        age = 0
        fixed_stop = np.nan
        horizon = max(1, min(2, int(params.holding_bars)))
        warmup = self.warmup_bars(params)
        for i in range(len(data)):
            # Yesterday's entry decision fills at today's open.
            if not active and i > 0 and decisions[i - 1] == 1:
                if np.isfinite(opens[i]) and opens[i] > 0 and np.isfinite(atr[i - 1]):
                    active = True
                    age = 0
                    fixed_stop = opens[i] - params.stop_atr * atr[i - 1]
            if active:
                age += 1
                # Freeze the entry stop; close-based exits fill next bar.
                if not np.isfinite(closes[i]) or closes[i] < fixed_stop or age >= horizon:
                    active = False
                    decisions[i] = 0
                else:
                    decisions[i] = 1
            elif i >= warmup and entry[i]:
                decisions[i] = 1
        df = pd.DataFrame({"signal": decisions, "size": 1.0}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")
