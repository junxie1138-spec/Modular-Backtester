from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class Params:
    seasonal_streak: int = 3
    atr_multiple: float = 1.25


class GeneratedStrategy(BaseStrategy[Params]):
    strategy_id = "gen_local_1791247376"

    @classmethod
    def params_type(cls) -> type[Params]:
        return Params

    def warmup_bars(self, params: Params) -> int:
        return 41

    def indicators(self, data: pd.DataFrame, params: Params) -> pd.DataFrame:
        out = pd.DataFrame(index=data.index)
        close = data["close"]
        previous_close = close.shift(1)
        out["return"] = close.pct_change(fill_method=None).replace([np.inf, -np.inf], np.nan)
        out["true_range"] = pd.concat([
            data["high"] - data["low"],
            (data["high"] - previous_close).abs(),
            (data["low"] - previous_close).abs(),
        ], axis=1).max(axis=1)
        out["atr"] = out["true_range"].rolling(14, min_periods=14).mean()
        out["weekday"] = data.index.dayofweek
        out["next_weekday"] = (out["weekday"] + 1) % 5
        for weekday in range(5):
            occurs = out["weekday"].eq(weekday)
            positive = occurs & out["return"].gt(0)
            reset = occurs & ~out["return"].gt(0)
            consecutive = positive.astype(int).groupby(reset.cumsum()).cumsum()
            recent_positive = positive.astype(float).rolling(40, min_periods=40).sum()
            out[f"streak_{weekday}"] = np.minimum(consecutive, recent_positive)
        out["expected_streak"] = np.nan
        for weekday in range(5):
            selected = out["next_weekday"].eq(weekday)
            out.loc[selected, "expected_streak"] = out.loc[selected, f"streak_{weekday}"]
        return out

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: Params,
    ) -> SignalFrame:
        if params.seasonal_streak < 1 or params.atr_multiple <= 0:
            raise ValueError("seasonal_streak and atr_multiple must be positive")
        close = data["close"].to_numpy(dtype=float)
        atr = indicators["atr"].to_numpy(dtype=float)
        eligible = indicators["expected_streak"].ge(params.seasonal_streak).to_numpy()
        raw = np.zeros(len(data), dtype=int)
        active = False
        held_sessions = 0
        high_water = np.nan
        trailing_stop = -np.inf
        for i in range(len(data)):
            valid = np.isfinite(close[i]) and np.isfinite(atr[i]) and atr[i] > 0
            if not valid:
                active = False
                held_sessions = 0
                high_water = np.nan
                trailing_stop = -np.inf
                continue
            if active:
                held_sessions += 1
                # Entry fills on this bar when held_sessions first reaches one.
                high_water = close[i] if held_sessions == 1 else max(high_water, close[i])
                trailing_stop = max(trailing_stop, high_water - params.atr_multiple * atr[i])
                if close[i] <= trailing_stop or held_sessions >= 2:
                    active = False
                    held_sessions = 0
                    high_water = np.nan
                    trailing_stop = -np.inf
                else:
                    raw[i] = 1
            elif eligible[i]:
                active = True
                held_sessions = 0
                high_water = np.nan
                trailing_stop = -np.inf
                raw[i] = 1
        df = pd.DataFrame({"signal": raw, "size": 1.0}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")
