from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class Params:
    streak_length: int = 2
    atr_period: int = 14
    trail_atr: float = 1.25
    hold_days: int = 2


class GeneratedStrategy(BaseStrategy[Params]):
    strategy_id = "gen_local_1791229962"

    @classmethod
    def params_type(cls) -> type[Params]:
        return Params

    @staticmethod
    def warmup_bars(params: Params) -> int:
        return int(max(200, params.atr_period + 1, params.streak_length + 1))

    @staticmethod
    def indicators(data: pd.DataFrame, params: Params) -> pd.DataFrame:
        if params.streak_length < 1 or params.atr_period < 1:
            raise ValueError("Lookbacks must be positive")
        if not np.isfinite(params.trail_atr) or params.trail_atr <= 0:
            raise ValueError("trail_atr must be finite and positive")
        if params.hold_days not in (1, 2):
            raise ValueError("hold_days must be one or two")
        out = pd.DataFrame(index=data.index)
        close = data["close"]
        previous = close.shift(1)
        out["change"] = close.diff()
        up = out["change"].gt(0)
        down = out["change"].lt(0)
        out["up_streak"] = up.groupby((~up).cumsum()).cumsum().clip(upper=params.streak_length)
        out["down_streak"] = down.groupby((~down).cumsum()).cumsum().clip(upper=params.streak_length)
        out["ma200"] = close.rolling(200, min_periods=200).mean()
        ranges = pd.concat([
            data["high"] - data["low"],
            (data["high"] - previous).abs(),
            (data["low"] - previous).abs(),
        ], axis=1)
        out["true_range"] = ranges.max(axis=1).where(previous.notna())
        out["atr"] = out["true_range"].rolling(params.atr_period, min_periods=params.atr_period).mean()
        day = data.index.day
        out["month_boundary"] = (day >= 25) | (day <= 3)
        out["midmonth"] = (day >= 10) & (day <= 20)
        ready = np.isfinite(close) & np.isfinite(out["ma200"]) & np.isfinite(out["atr"]) & out["atr"].gt(0)
        buy = ready & out["month_boundary"] & close.gt(out["ma200"]) & out["down_streak"].ge(params.streak_length)
        sell = ready & out["midmonth"] & close.lt(out["ma200"]) & out["up_streak"].ge(params.streak_length)
        out["entry"] = np.select([buy, sell], [1, -1], default=0).astype(int)
        return out

    @staticmethod
    def generate_signals(data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: Params) -> SignalFrame:
        if params.hold_days not in (1, 2) or not np.isfinite(params.trail_atr) or params.trail_atr <= 0:
            raise ValueError("Invalid holding period or trailing distance")
        signals = np.zeros(len(data), dtype=int)
        position = 0
        age = 0
        highest = -np.inf
        lowest = np.inf
        stop = -np.inf
        short_stop = np.inf
        closes = data["close"].to_numpy(dtype=float)
        atrs = indicators["atr"].to_numpy(dtype=float)
        entries = indicators["entry"].fillna(0).to_numpy(dtype=int)
        for i in range(len(data)):
            close, atr = closes[i], atrs[i]
            if not np.isfinite(close) or not np.isfinite(atr) or atr <= 0:
                position = 0
                signals[i] = 0
                continue
            if position:
                # The prior decision filled this morning; count only closes held in trade.
                age += 1
                highest = max(highest, close)
                lowest = min(lowest, close)
                stop = max(stop, highest - params.trail_atr * atr)
                short_stop = min(short_stop, lowest + params.trail_atr * atr)
                trailing_exit = close < stop
                short_adverse_exit = position == -1 and close > short_stop
                if trailing_exit or short_adverse_exit or age >= params.hold_days:
                    position = 0
            elif entries[i] in (-1, 1):
                position = int(entries[i])
                age = 0
                highest, lowest = -np.inf, np.inf
                stop, short_stop = -np.inf, np.inf
            signals[i] = position
        df = pd.DataFrame({"signal": signals, "size": 1.0}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check() -> None:
    index = pd.date_range("2024-01-01", periods=5, freq="B")
    data = pd.DataFrame({"close": [100.0, 102.0, 99.0, 100.0, 100.0]}, index=index)
    for direction in (1, -1):
        indicators = pd.DataFrame({"atr": 1.0, "entry": [direction, 0, 0, 0, 0]}, index=index)
        for days in (1, 2):
            result = GeneratedStrategy.generate_signals(data, indicators, None, Params(hold_days=days)).data
            expected = [0, direction, direction if days == 2 else 0, 0, 0]
            assert result["signal"].tolist() == expected
            assert result["size"].gt(0).all()


if __name__ == "__main__":
    _self_check()
