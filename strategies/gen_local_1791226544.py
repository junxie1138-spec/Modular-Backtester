from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class StreakRegimeParams:
    atr_window: int = 14
    regime_window: int = 40
    stretch_streak: int = 4
    trail_atr: float = 2.5
    max_hold: int = 20


class GeneratedStrategy(BaseStrategy[StreakRegimeParams]):
    strategy_id = "gen_local_1791226544"

    @classmethod
    def params_type(cls) -> type[StreakRegimeParams]:
        return StreakRegimeParams

    @classmethod
    def warmup_bars(cls, params: StreakRegimeParams) -> int:
        return params.atr_window + params.regime_window

    def indicators(self, data: pd.DataFrame, params: StreakRegimeParams) -> pd.DataFrame:
        close = data["close"]
        previous_close = close.shift(1)
        true_range = pd.concat(
            [data["high"] - data["low"],
             (data["high"] - previous_close).abs(),
             (data["low"] - previous_close).abs()],
            axis=1,
        ).max(axis=1)
        atr = true_range.rolling(params.atr_window, min_periods=params.atr_window).mean()
        atr_pct = atr / close.replace(0, np.nan)
        baseline = atr_pct.rolling(params.regime_window, min_periods=params.regime_window).median()
        up = close.diff().gt(0)
        down = close.diff().lt(0)
        return pd.DataFrame({
            "atr": atr.fillna(0.0),
            "valid": baseline.notna() & atr.gt(0),
            "stressed": atr_pct.gt(baseline).fillna(False),
            "up_streak": up.groupby((~up).cumsum()).cumsum().fillna(0).astype(int),
            "down_streak": down.groupby((~down).cumsum()).cumsum().fillna(0).astype(int),
        }, index=data.index)

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: StreakRegimeParams,
    ) -> SignalFrame:
        valid = indicators["valid"]
        stressed = indicators["stressed"]
        calm_pair = valid & ~stressed & (~stressed).shift(1).fillna(False)
        stressed_pair = valid & stressed & stressed.shift(1).fillna(False)
        up = indicators["up_streak"]
        down = indicators["down_streak"]
        long_entry = (calm_pair & up.eq(2)) | (
            stressed_pair & down.shift(2).ge(params.stretch_streak) & up.eq(2)
        )
        short_entry = (calm_pair & down.eq(2)) | (
            stressed_pair & up.shift(2).ge(params.stretch_streak) & down.eq(2)
        )

        closes = data["close"].to_numpy(dtype=float)
        atrs = indicators["atr"].to_numpy(dtype=float)
        longs = long_entry.fillna(False).to_numpy(dtype=bool)
        shorts = short_entry.fillna(False).to_numpy(dtype=bool)
        positions = np.zeros(len(data), dtype=int)
        position = 0
        age = 0
        extreme = 0.0
        stop = 0.0
        for i, close in enumerate(closes):
            atr = atrs[i]
            if position == 1:
                age += 1
                extreme = max(extreme, close)
                stop = max(stop, extreme - params.trail_atr * atr)
                if close <= stop or age >= params.max_hold:
                    position = 0
            elif position == -1:
                age += 1
                extreme = min(extreme, close)
                stop = min(stop, extreme + params.trail_atr * atr)
                if close >= stop or age >= params.max_hold:
                    position = 0
            elif atr > 0 and (longs[i] or shorts[i]):
                position = 1 if longs[i] else -1
                age = 0
                extreme = close
                stop = close - params.trail_atr * atr if position == 1 else close + params.trail_atr * atr
            positions[i] = position

        df = pd.DataFrame({"signal": positions, "size": 1.0}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")
