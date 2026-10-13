from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class StreakParams:
    history: int = 126
    min_streak: int = 3
    persistence_edge: float = 0.08
    atr_window: int = 14
    atr_multiple: float = 3.0
    holding_bars: int = 18

    def __post_init__(self):
        if min(self.history, self.min_streak, self.atr_window) < 2:
            raise ValueError("Lookbacks and streak threshold must be at least two.")
        if self.holding_bars < 1 or not 0.0 <= self.persistence_edge <= 1.0:
            raise ValueError("Invalid holding period or persistence edge.")
        if not np.isfinite(self.atr_multiple) or self.atr_multiple <= 0:
            raise ValueError("ATR multiple must be finite and positive.")


class GeneratedStrategy(BaseStrategy[StreakParams]):
    strategy_id = "gen_local-w2_1791342918"

    @classmethod
    def params_type(cls):
        return StreakParams

    def warmup_bars(self, params: StreakParams) -> int:
        return int(max(params.history + params.min_streak + 2, params.atr_window + 1))

    def indicators(self, data: pd.DataFrame, params: StreakParams) -> pd.DataFrame:
        out = pd.DataFrame(index=data.index)
        close = data["close"]
        change = close.diff()
        direction = np.sign(change).fillna(0).astype(int)
        groups = direction.ne(direction.shift()).cumsum()
        streak = (direction.groupby(groups).cumcount() + 1).clip(upper=params.min_streak)
        streak = streak.where(direction.ne(0), 0)
        previous_direction = direction.shift()
        previous_streak = streak.shift()
        continued = direction.eq(previous_direction) & direction.ne(0)
        late = previous_streak.ge(params.min_streak) & direction.ne(0)
        early = previous_streak.eq(1) & direction.ne(0)
        # Exclude today's transition from the estimated continuation rates.
        late_count = late.astype(float).shift(1).rolling(params.history).sum()
        early_count = early.astype(float).shift(1).rolling(params.history).sum()
        late_hits = (late & continued).astype(float).shift(1).rolling(params.history).sum()
        early_hits = (early & continued).astype(float).shift(1).rolling(params.history).sum()
        late_rate = late_hits.div(late_count.replace(0, np.nan))
        early_rate = early_hits.div(early_count.replace(0, np.nan))
        previous_close = close.shift()
        true_range = pd.concat([
            data["high"] - data["low"],
            (data["high"] - previous_close).abs(),
            (data["low"] - previous_close).abs(),
        ], axis=1).max(axis=1)
        true_range = true_range.where(previous_close.notna())
        atr = true_range.rolling(params.atr_window).mean()
        ready = (
            late_count.ge(8) & early_count.ge(12)
            & late_rate.gt(0.5)
            & (late_rate - early_rate).ge(params.persistence_edge)
            & streak.ge(params.min_streak) & atr.gt(0)
        ).fillna(False)
        ready.iloc[:self.warmup_bars(params)] = False
        out["direction"] = direction
        out["streak"] = streak
        out["late_count"] = late_count
        out["early_count"] = early_count
        out["late_rate"] = late_rate
        out["early_rate"] = early_rate
        out["atr"] = atr
        out["candidate"] = direction.where(ready, 0).astype(int)
        return out

    def generate_signals(
        self, data: pd.DataFrame, indicators: pd.DataFrame,
        ctx: StrategyContext, params: StreakParams,
    ) -> SignalFrame:
        close = data["close"].to_numpy(dtype=float)
        opening = data["open"].to_numpy(dtype=float)
        atr = indicators["atr"].to_numpy(dtype=float)
        candidate = indicators["candidate"].fillna(0).to_numpy(dtype=int)
        signal = np.zeros(len(data), dtype=int)
        position = pending = age = 0
        highest = stop = -np.inf
        entry_atr = short_limit = np.nan
        for i in range(len(data)):
            if not np.isfinite(close[i]) or not np.isfinite(atr[i]) or atr[i] <= 0:
                position = pending = age = 0
                highest = stop = -np.inf
                continue
            if position:
                age += 1
                if age == 1 and position == -1:
                    short_limit = opening[i] + params.atr_multiple * entry_atr
                highest = max(highest, close[i])
                # The required high-water stop only ratchets upward, for either side.
                stop = max(stop, highest - params.atr_multiple * atr[i])
                # Shorts also have a fixed adverse-move cap from the actual entry open.
                exit_now = (
                    close[i] <= stop or age >= params.holding_bars
                    or (position == -1 and close[i] >= short_limit)
                )
                if exit_now:
                    position = pending = age = 0
                    highest = stop = -np.inf
            else:
                wanted = int(candidate[i])
                if wanted != 0 and wanted == pending:
                    position = wanted
                    age = pending = 0
                    highest = stop = -np.inf
                    entry_atr = atr[i]
                    short_limit = np.nan
                else:
                    pending = wanted
            signal[i] = position
        df = pd.DataFrame({"signal": signal, "size": 0.95}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def self_check():
    index = pd.date_range("2020-01-01", periods=9)
    close = np.array([100, 101, 103, 102, 99, 100, 99, 104, 103], dtype=float)
    data = pd.DataFrame({
        "open": close, "high": close + 1, "low": close - 1,
        "close": close, "volume": 1000.0,
    }, index=index)
    data.loc[index[7], "open"] = 99.0
    derived = pd.DataFrame({
        "atr": 1.0, "candidate": [1, 1, 1, 1, 1, -1, -1, -1, -1],
    }, index=index)
    strategy = GeneratedStrategy()
    params = StreakParams(atr_multiple=2.0)
    result = strategy.generate_signals(data, derived, None, params).data
    assert result["signal"].tolist() == [0, 0, 1, 1, 1, 0, 0, -1, 0]
    assert (result["size"] > 0).all()
    computed = strategy.indicators(data, params)
    assert computed.index.equals(data.index)
    assert computed["candidate"].eq(0).all()


if __name__ == "__main__":
    self_check()
