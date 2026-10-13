from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class StreakParams:
    atr_period: int = 14
    min_streak: int = 2
    retention_min: float = 0.55
    trail_atr: float = 0.9
    hold_bars: int = 2


class GeneratedStrategy(BaseStrategy[StreakParams]):
    strategy_id = "gen_local_1791213219"

    @classmethod
    def params_type(cls):
        return StreakParams

    @staticmethod
    def warmup_bars(params: StreakParams) -> int:
        if params.atr_period < 1 or params.min_streak < 1:
            raise ValueError("Lookbacks must be positive")
        if not 0.0 < params.retention_min <= 1.0:
            raise ValueError("retention_min must be in (0, 1]")
        if not np.isfinite(params.trail_atr) or params.trail_atr <= 0:
            raise ValueError("trail_atr must be finite and positive")
        if params.hold_bars not in (1, 2):
            raise ValueError("hold_bars must be 1 or 2")
        return int(max(params.atr_period + 1, params.min_streak + 2))

    @staticmethod
    def indicators(data: pd.DataFrame, params: StreakParams) -> pd.DataFrame:
        close = data["close"]
        previous = close.shift(1)
        change = close.diff()
        direction = pd.Series(np.sign(change), index=data.index).fillna(0).astype(int)
        groups = direction.ne(direction.shift(1)).cumsum()
        streak = direction.groupby(groups).cumcount().add(1).where(direction.ne(0), 0)
        true_range = pd.concat([
            data["high"] - data["low"],
            (data["high"] - previous).abs(),
            (data["low"] - previous).abs(),
        ], axis=1).max(axis=1)
        true_range = true_range.where(previous.notna())
        atr = true_range.rolling(params.atr_period, min_periods=params.atr_period).mean()
        retention = direction * (close - data["open"]) / true_range.replace(0, np.nan)
        valid = np.isfinite(data[["open", "high", "low", "close"]]).all(axis=1)
        valid &= np.isfinite(atr) & atr.gt(0) & np.isfinite(retention)
        candidate = direction.where(
            valid
            & streak.ge(params.min_streak)
            & retention.ge(params.retention_min)
            & retention.gt(retention.shift(1))
            & direction.eq(direction.shift(1)),
            0,
        ).astype(int)
        confirmed = candidate.where(candidate.eq(candidate.shift(1)), 0).astype(int)
        return pd.DataFrame({
            "change": change,
            "direction": direction,
            "streak": streak,
            "true_range": true_range,
            "atr": atr,
            "retention": retention,
            "candidate": candidate,
            "confirmed": confirmed,
        }, index=data.index)

    @staticmethod
    def generate_signals(
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: StreakParams,
    ) -> SignalFrame:
        close = data["close"].to_numpy(dtype=float)
        opening = data["open"].to_numpy(dtype=float)
        atr = indicators["atr"].to_numpy(dtype=float)
        entries = indicators["confirmed"].to_numpy(dtype=int)
        signals = np.zeros(len(data), dtype=int)
        position = 0
        age = 0
        watermark = np.nan
        stop = np.nan

        for i in range(len(data)):
            valid = np.isfinite(close[i]) and np.isfinite(atr[i]) and atr[i] > 0
            if position:
                age += 1
                if not valid or not np.isfinite(opening[i]):
                    position = 0
                else:
                    if age == 1:
                        watermark = opening[i]
                        stop = opening[i] - position * params.trail_atr * atr[i - 1]
                    if position == 1:
                        watermark = max(watermark, close[i])
                        stop = max(stop, watermark - params.trail_atr * atr[i])
                        stopped = close[i] <= stop
                    else:
                        watermark = min(watermark, close[i])
                        stop = min(stop, watermark + params.trail_atr * atr[i])
                        stopped = close[i] >= stop
                    if stopped or age >= params.hold_bars:
                        position = 0
            elif valid and entries[i] != 0:
                position = int(entries[i])
                age = 0
                watermark = np.nan
                stop = np.nan
            signals[i] = position

        df = pd.DataFrame({"signal": signals, "size": 1.0}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check() -> None:
    index = pd.date_range("2020-01-01", periods=6)
    data = pd.DataFrame({
        "open": [100.0, 100.0, 101.0, 102.0, 101.0, 100.0],
        "high": [101.0, 102.0, 103.0, 103.0, 102.0, 101.0],
        "low": [99.0, 99.0, 100.0, 100.0, 99.0, 98.0],
        "close": [100.0, 101.0, 102.0, 101.0, 100.0, 99.0],
        "volume": [1000.0] * 6,
    }, index=index)
    derived = pd.DataFrame({
        "atr": [1.0] * 6,
        "confirmed": [0, 1, 0, 0, -1, 0],
    }, index=index)
    result = GeneratedStrategy.generate_signals(data, derived, None, StreakParams()).data
    assert result["signal"].tolist() == [0, 0, 1, 1, 0, -1]
    assert result["size"].gt(0).all()
    falling = data.copy()
    falling.loc[index[2], "close"] = 99.0
    stopped = GeneratedStrategy.generate_signals(falling, derived, None, StreakParams()).data
    assert stopped["signal"].iloc[3] == 0
    actual = GeneratedStrategy.indicators(data, StreakParams(atr_period=2))
    assert actual.index.equals(data.index)
    assert actual["confirmed"].isin([-1, 0, 1]).all()


if __name__ == "__main__":
    _self_check()
