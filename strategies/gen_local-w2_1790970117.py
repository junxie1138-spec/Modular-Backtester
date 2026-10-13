from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class StreakParams:
    streak_length: int = 3
    atr_multiple: float = 2.5


class GeneratedStrategy(BaseStrategy[StreakParams]):
    strategy_id = "gen_local-w2_1790970117"

    @classmethod
    def params_type(cls):
        return StreakParams

    def warmup_bars(self, params: StreakParams) -> int:
        return 128 + params.streak_length

    def indicators(self, data: pd.DataFrame, params: StreakParams) -> pd.DataFrame:
        if params.streak_length < 2:
            raise ValueError("streak_length must be at least two")
        if not np.isfinite(params.atr_multiple) or params.atr_multiple <= 0:
            raise ValueError("atr_multiple must be finite and positive")
        out = pd.DataFrame(index=data.index)
        close = data["close"]
        out["change"] = close.diff()
        out["direction"] = np.sign(out["change"]).fillna(0).astype(int)
        direction = out["direction"]
        groups = direction.ne(direction.shift()).cumsum()
        out["streak"] = direction.ne(0).astype(int).groupby(groups).cumsum()
        previous_close = close.shift(1)
        out["true_range"] = pd.concat([
            data["high"] - data["low"],
            (data["high"] - previous_close).abs(),
            (data["low"] - previous_close).abs(),
        ], axis=1).max(axis=1)
        out["atr"] = out["true_range"].rolling(14, min_periods=14).mean()
        out["opportunity"] = direction.shift(1).fillna(0).ne(0).astype(float)
        out["continued"] = direction.eq(direction.shift(1)).astype(float) * out["opportunity"]
        out["eligible"] = (out["streak"].shift(1) >= params.streak_length).astype(float)
        out["eligible_continued"] = out["eligible"] * out["continued"]
        out["samples"] = out["eligible"].rolling(126, min_periods=126).sum()
        out["conditional_probability"] = (
            out["eligible_continued"].rolling(126, min_periods=126).sum()
            / out["samples"].replace(0, np.nan)
        )
        out["baseline_probability"] = (
            out["continued"].rolling(126, min_periods=126).sum()
            / out["opportunity"].rolling(126, min_periods=126).sum().replace(0, np.nan)
        )
        probability = out["conditional_probability"]
        baseline = out["baseline_probability"]
        fresh = out["streak"].eq(params.streak_length) & out["samples"].ge(8)
        follow = fresh & probability.gt(0.55) & probability.gt(baseline + 0.08)
        fade = fresh & probability.lt(0.45) & probability.lt(baseline - 0.08)
        out["entry"] = np.select([follow, fade], [direction, -direction], default=0).astype(int)
        out.loc[~np.isfinite(out["atr"]) | out["atr"].le(0), "entry"] = 0
        out.iloc[:self.warmup_bars(params), out.columns.get_loc("entry")] = 0
        return out

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: StreakParams,
    ) -> SignalFrame:
        df = pd.DataFrame(index=data.index)
        signals = np.zeros(len(data), dtype=int)
        position = 0
        entry_bar = 0
        highest = lowest = high_stop = short_stop = 0.0
        closes = data["close"].to_numpy(dtype=float)
        atrs = indicators["atr"].to_numpy(dtype=float)
        entries = indicators["entry"].to_numpy(dtype=int)
        for i, (close, atr, entry) in enumerate(zip(closes, atrs, entries)):
            if not np.isfinite(close) or not np.isfinite(atr) or atr <= 0:
                signals[i] = position
                continue
            distance = params.atr_multiple * atr
            if position:
                highest = max(highest, close)
                lowest = min(lowest, close)
                # The required highest-close stop only ratchets upward.
                high_stop = max(high_stop, highest - distance)
                # Shorts additionally receive a mirrored adverse-move stop.
                short_stop = min(short_stop, lowest + distance)
                if close < high_stop or (position < 0 and close > short_stop) or i - entry_bar >= 8:
                    position = 0
                signals[i] = position
                continue
            if entry:
                position = int(entry)
                entry_bar = i
                highest = lowest = close
                high_stop = close - distance
                short_stop = close + distance
            signals[i] = position
        df["signal"] = signals
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = 1.0
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check():
    strategy = GeneratedStrategy()
    params = StreakParams()
    index = pd.date_range("2020-01-01", periods=12)
    data = pd.DataFrame({"close": [100, 103, 105, 104, 102, 100, 100, 104, 104, 104, 104, 104]}, index=index)
    indicators = pd.DataFrame({"atr": [1, 1, 1, 10, 10, 1, 1, 1, 1, 1, 1, 1], "entry": [1, 0, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0]}, index=index)
    result = strategy.generate_signals(data, indicators, None, params).data
    assert result["signal"].tolist() == [0, 1, 1, 1, 1, 0, 0, -1, 0, 0, 0, 0]
    indicators["atr"] = 10.0
    indicators["entry"] = 0
    indicators.iloc[0, indicators.columns.get_loc("entry")] = 1
    result = strategy.generate_signals(data, indicators, None, params).data
    assert result["signal"].tolist() == [0] + [1] * 8 + [0] * 3
    assert result["size"].gt(0).all()


if __name__ == "__main__":
    _self_check()
