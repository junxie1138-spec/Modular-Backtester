from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class StaircaseParams:
    atr_period: int = 14
    min_streak: int = 4
    streak_cap: int = 8
    min_close_location: float = 0.65
    hold_bars: int = 8
    stop_atr: float = 2.0


class GeneratedStrategy(BaseStrategy[StaircaseParams]):
    strategy_id = "gen_local_1791258633"

    @classmethod
    def params_type(cls) -> type[StaircaseParams]:
        return StaircaseParams

    @staticmethod
    def warmup_bars(params: StaircaseParams) -> int:
        return int(max(params.atr_period, params.streak_cap, params.min_streak) + 1)

    @staticmethod
    def indicators(data: pd.DataFrame, params: StaircaseParams) -> pd.DataFrame:
        if not (params.atr_period >= 1 and 1 <= params.min_streak <= params.streak_cap and params.hold_bars >= 1 and np.isfinite(params.stop_atr) and params.stop_atr > 0 and 0 <= params.min_close_location <= 1):
            raise ValueError("Invalid staircase parameters")
        close = data["close"]
        previous_close = close.shift(1)
        true_range = pd.concat([
            data["high"] - data["low"],
            (data["high"] - previous_close).abs(),
            (data["low"] - previous_close).abs(),
        ], axis=1).max(axis=1)
        atr = true_range.rolling(params.atr_period, min_periods=params.atr_period).mean()
        rising_low = data["low"].gt(data["low"].shift(1)).fillna(False)
        streak = rising_low.astype(int).groupby((~rising_low).cumsum()).cumsum().clip(upper=params.streak_cap)
        daily_range = (data["high"] - data["low"]).replace(0.0, np.nan)
        close_location = ((close - data["low"]) / daily_range).clip(0.0, 1.0).fillna(0.0)
        progress = (close - close.shift(params.min_streak)).fillna(0.0)
        strength = (0.5 * streak / params.streak_cap + 0.5 * close_location).clip(0.0, 1.0)
        return pd.DataFrame({
            "atr": atr,
            "streak": streak,
            "close_location": close_location,
            "progress": progress,
            "strength": strength,
        }, index=data.index)

    @staticmethod
    def generate_signals(data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: StaircaseParams) -> SignalFrame:
        entry = (
            indicators["streak"].ge(params.min_streak)
            & indicators["close_location"].ge(params.min_close_location)
            & indicators["progress"].gt(0.0)
            & indicators["atr"].gt(0.0)
            & np.isfinite(indicators["atr"])
            & np.isfinite(data["close"])
        ).fillna(False).to_numpy(dtype=bool)
        opens = data["open"].to_numpy(dtype=float)
        closes = data["close"].to_numpy(dtype=float)
        atrs = indicators["atr"].to_numpy(dtype=float)
        strengths = indicators["strength"].fillna(0.0).clip(0.0, 1.0).to_numpy(dtype=float)
        signals = np.zeros(len(data), dtype=int)
        sizes = np.ones(len(data), dtype=float)
        active = False
        pending_fill = False
        stop_price = 0.0
        entry_atr = 0.0
        age = 0
        position_size = 1.0
        for i in range(len(data)):
            if pending_fill:
                pending_fill = False
                if not np.isfinite(opens[i]) or opens[i] <= 0.0:
                    active = False
                    continue
                stop_price = opens[i] - params.stop_atr * entry_atr
                age = 0
            if active:
                age += 1
                if not np.isfinite(closes[i]) or closes[i] < stop_price or age >= params.hold_bars:
                    active = False
                    continue
                signals[i] = 1
                sizes[i] = position_size
            elif entry[i]:
                active = True
                pending_fill = True
                entry_atr = atrs[i]
                position_size = float(0.25 + 0.75 * strengths[i])
                signals[i] = 1
                sizes[i] = position_size
        df = pd.DataFrame({"signal": signals, "size": sizes}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = df["size"].shift(1).fillna(1.0).astype(float)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check() -> None:
    index = pd.date_range("2020-01-01", periods=6)
    data = pd.DataFrame({
        "open": [90.0, 100.0, 99.0, 98.0, 95.0, 95.0],
        "high": [101.0] * 6,
        "low": [89.0] * 6,
        "close": [90.0, 99.0, 98.0, 95.0, 95.0, 95.0],
        "volume": [1000.0] * 6,
    }, index=index)
    indicators = pd.DataFrame({
        "atr": [2.0] * 6,
        "streak": [4, 0, 0, 0, 0, 0],
        "close_location": [0.9] * 6,
        "progress": [1.0] * 6,
        "strength": [0.7] * 6,
    }, index=index)
    result = GeneratedStrategy.generate_signals(data, indicators, None, StaircaseParams()).data
    assert result["signal"].tolist() == [0, 1, 1, 1, 0, 0]
    assert np.isclose(result["size"].iloc[1], 0.775)
    assert result["size"].gt(0.0).all()
    expiry = GeneratedStrategy.generate_signals(data, indicators, None, StaircaseParams(hold_bars=2)).data
    assert expiry["signal"].tolist() == [0, 1, 1, 0, 0, 0]


if __name__ == "__main__":
    _self_check()
