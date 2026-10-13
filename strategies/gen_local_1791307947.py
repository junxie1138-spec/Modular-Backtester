from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class StreakParams:
    percentile_window: int = 252
    streak_percentile: float = 0.90
    atr_window: int = 14
    stop_atr: float = 2.5
    hold_bars: int = 18


class GeneratedStrategy(BaseStrategy[StreakParams]):
    strategy_id = "gen_local_1791307947"

    @classmethod
    def params_type(cls):
        return StreakParams

    def warmup_bars(self, params: StreakParams) -> int:
        return max(params.percentile_window + 22, params.atr_window + 1)

    def indicators(self, data: pd.DataFrame, params: StreakParams) -> pd.DataFrame:
        if (params.percentile_window < 2 or params.atr_window < 1
                or params.hold_bars < 1 or not 0 < params.streak_percentile < 1
                or not np.isfinite(params.stop_atr) or params.stop_atr <= 0):
            raise ValueError("Invalid strategy parameters")
        close = data["close"]
        change = close.diff()
        down = change.lt(0)
        # Capping makes the streak depend on at most twenty daily changes.
        streak = down.astype(int).groupby((~down).cumsum()).cumsum().clip(upper=20)
        streak = streak.astype(float).where(change.notna())
        threshold = streak.shift(1).rolling(
            params.percentile_window, min_periods=params.percentile_window
        ).quantile(params.streak_percentile)
        previous_close = close.shift(1)
        true_range = pd.concat([
            data["high"] - data["low"],
            (data["high"] - previous_close).abs(),
            (data["low"] - previous_close).abs(),
        ], axis=1).max(axis=1, skipna=False)
        atr = true_range.rolling(
            params.atr_window, min_periods=params.atr_window
        ).mean()
        entry = (
            change.gt(0)
            & streak.shift(1).gt(threshold.shift(1))
            & atr.gt(0)
            & np.isfinite(atr)
            & close.gt(0)
        ).fillna(False)
        return pd.DataFrame({
            "down_streak": streak,
            "streak_threshold": threshold,
            "atr": atr,
            "entry": entry,
        }, index=data.index)

    def generate_signals(
        self, data: pd.DataFrame, indicators: pd.DataFrame,
        ctx: StrategyContext, params: StreakParams,
    ) -> SignalFrame:
        entries = indicators["entry"].fillna(False).to_numpy(dtype=bool)
        atr = indicators["atr"].to_numpy(dtype=float)
        opens = data["open"].to_numpy(dtype=float)
        closes = data["close"].to_numpy(dtype=float)
        signal = np.zeros(len(data), dtype=int)
        active = False
        entry_bar = -1
        entry_atr = np.nan
        stop = np.nan
        for i in range(len(data)):
            if active:
                if i == entry_bar:
                    # Entry is executed at this open; ATR was known at the prior close.
                    stop = opens[i] - params.stop_atr * entry_atr
                invalid = not np.isfinite(closes[i]) or not np.isfinite(stop)
                if invalid or closes[i] < stop or i - entry_bar + 1 >= params.hold_bars:
                    active = False
            elif entries[i] and np.isfinite(atr[i]) and atr[i] > 0:
                active = True
                entry_bar = i + 1
                entry_atr = atr[i]
            signal[i] = int(active)
        df = pd.DataFrame({"signal": signal, "size": 0.95}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


if __name__ == "__main__":
    strategy = GeneratedStrategy()
    index = pd.date_range("2020-01-01", periods=6)
    sample = pd.DataFrame({
        "open": [100., 90., 91., 110., 87., 85.],
        "high": [101., 92., 111., 111., 88., 91.],
        "low": [99., 89., 90., 86., 84., 84.],
        "close": [100., 91., 110., 87., 83., 90.],
        "volume": [1000.] * 6,
    }, index=index)
    derived = pd.DataFrame({
        "entry": [True, False, False, False, False, False],
        "atr": [2.] * 6,
    }, index=index)
    result = strategy.generate_signals(sample, derived, None, StreakParams())
    assert result.data["signal"].tolist() == [0, 1, 1, 1, 1, 0]
    timed = strategy.generate_signals(sample, derived, None, StreakParams(hold_bars=3))
    assert timed.data["signal"].tolist() == [0, 1, 1, 1, 0, 0]
