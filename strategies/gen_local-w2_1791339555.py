from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class SeasonalStreakParams:
    entry_weekday: int = 3
    min_streak: int = 2
    streak_cap: int = 6
    close_location_min: float = 0.55
    atr_period: int = 14
    breakeven_pct: float = 0.01
    trail_atr: float = 1.5
    initial_stop_atr: float = 2.0
    hold_days: int = 4


class GeneratedStrategy(BaseStrategy[SeasonalStreakParams]):
    strategy_id = "gen_local-w2_1791339555"

    @classmethod
    def params_type(cls) -> type[SeasonalStreakParams]:
        return SeasonalStreakParams

    @staticmethod
    def warmup_bars(params: SeasonalStreakParams) -> int:
        return int(max(params.atr_period + 1, params.streak_cap))

    @staticmethod
    def indicators(data: pd.DataFrame, params: SeasonalStreakParams) -> pd.DataFrame:
        if not (0 <= params.entry_weekday <= 4 and 1 <= params.min_streak <= params.streak_cap):
            raise ValueError("Invalid weekday or streak parameters")
        if not (params.atr_period >= 1 and 3 <= params.hold_days <= 5):
            raise ValueError("Invalid ATR period or holding horizon")
        if not (0 <= params.close_location_min <= 1 and params.breakeven_pct > 0 and params.trail_atr > 0 and params.initial_stop_atr > 0):
            raise ValueError("Invalid stop or close-location parameters")
        out = pd.DataFrame(index=data.index)
        out["intraday_return"] = data["close"].div(data["open"].replace(0, np.nan)).sub(1)
        selling = out["intraday_return"].lt(0)
        groups = (~selling).cumsum()
        out["selling_streak"] = selling.astype(int).groupby(groups).cumsum().clip(upper=params.streak_cap)
        out["previous_close"] = data["close"].shift(1)
        out["true_range"] = pd.concat([
            data["high"] - data["low"],
            (data["high"] - out["previous_close"]).abs(),
            (data["low"] - out["previous_close"]).abs(),
        ], axis=1).max(axis=1)
        out["atr"] = out["true_range"].rolling(params.atr_period, min_periods=params.atr_period).mean()
        out["close_location"] = (data["close"] - data["low"]).div((data["high"] - data["low"]).replace(0, np.nan))
        out["weekday"] = data.index.dayofweek
        out["strength"] = (0.35 + 0.65 * out["selling_streak"] / params.streak_cap).clip(0.35, 1.0)
        out["entry_candidate"] = (
            out["weekday"].eq(params.entry_weekday)
            & out["selling_streak"].ge(params.min_streak)
            & out["close_location"].ge(params.close_location_min)
            & out["atr"].gt(0)
            & np.isfinite(out["atr"])
        ).fillna(False)
        return out

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: SeasonalStreakParams) -> SignalFrame:
        n = len(data)
        signals = np.zeros(n, dtype=int)
        sizes = np.ones(n, dtype=float)
        active = False
        pending = False
        armed = False
        entry = 0.0
        stop = 0.0
        peak = 0.0
        entry_atr = 0.0
        allocation = 1.0
        age = 0
        for i in range(n):
            opening = float(data["open"].iloc[i])
            high = float(data["high"].iloc[i])
            low = float(data["low"].iloc[i])
            close = float(data["close"].iloc[i])
            atr = float(indicators["atr"].iloc[i])
            if pending:
                pending = False
                if np.isfinite(opening) and opening > 0:
                    entry = opening
                    stop = entry - params.initial_stop_atr * entry_atr
                    peak = entry
                    armed = False
                    age = 0
                    active = True
            if active:
                age += 1
                # Stops observed during this session produce a next-open exit.
                breached = (np.isfinite(low) and low <= stop) or (np.isfinite(close) and close <= stop)
                if breached or age >= params.hold_days:
                    active = False
                    sizes[i] = allocation
                    continue
                if np.isfinite(high):
                    peak = max(peak, high)
                    if high >= entry * (1.0 + params.breakeven_pct):
                        armed = True
                if armed:
                    stop = max(stop, entry)
                    if np.isfinite(atr) and atr > 0:
                        stop = max(stop, peak - params.trail_atr * atr)
                    if np.isfinite(close) and close <= stop:
                        active = False
                        sizes[i] = allocation
                        continue
                signals[i] = 1
                sizes[i] = allocation
            elif bool(indicators["entry_candidate"].iloc[i]) and np.isfinite(atr) and atr > 0:
                allocation = float(np.clip(indicators["strength"].iloc[i], 0.35, 1.0))
                entry_atr = atr
                pending = True
                signals[i] = 1
                sizes[i] = allocation
        df = pd.DataFrame({"signal": signals, "size": sizes}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = df["size"].shift(1).fillna(1.0).astype(float)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


if __name__ == "__main__":
    index = pd.bdate_range("2024-01-04", periods=5)
    data = pd.DataFrame({
        "open": [100.0] * 5,
        "high": [100.0, 102.0, 103.0, 104.0, 104.0],
        "low": [99.0, 99.0, 100.0, 100.0, 102.0],
        "close": [100.0, 101.5, 102.0, 103.0, 103.0],
        "volume": [1000.0] * 5,
    }, index=index)
    indicators = pd.DataFrame({
        "atr": [1.0] * 5,
        "strength": [0.8] * 5,
        "entry_candidate": [True, False, False, False, False],
    }, index=index)
    strategy = GeneratedStrategy.__new__(GeneratedStrategy)
    result = strategy.generate_signals(data, indicators, None, SeasonalStreakParams()).data
    assert result["signal"].tolist() == [0, 1, 1, 0, 0]
    assert np.allclose(result.loc[result["signal"].eq(1), "size"], 0.8)
    assert result["size"].gt(0).all()
