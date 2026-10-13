from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class GapRejectionParams:
    streak_bars: int = 3
    min_gap_atr: float = 0.1
    atr_bars: int = 14
    trailing_atr: float = 2.5
    max_hold_bars: int = 10


class GeneratedStrategy(BaseStrategy[GapRejectionParams]):
    strategy_id = "gen_local_1791261089"

    @classmethod
    def params_type(cls):
        return GapRejectionParams

    def warmup_bars(self, params):
        return int(max(200, params.atr_bars + params.streak_bars + 1))

    def indicators(self, data, params):
        if params.streak_bars < 1 or params.atr_bars < 1 or params.max_hold_bars < 1:
            raise ValueError("Bar counts must be positive")
        if not np.isfinite(params.min_gap_atr) or params.min_gap_atr < 0:
            raise ValueError("min_gap_atr must be finite and nonnegative")
        if not np.isfinite(params.trailing_atr) or params.trailing_atr <= 0:
            raise ValueError("trailing_atr must be finite and positive")
        prices = data[["open", "high", "low", "close"]].astype(float).replace([np.inf, -np.inf], np.nan)
        close = prices["close"]
        previous_close = close.shift(1)
        true_range = pd.concat([
            prices["high"] - prices["low"],
            (prices["high"] - previous_close).abs(),
            (prices["low"] - previous_close).abs(),
        ], axis=1).max(axis=1, skipna=False)
        atr = true_range.rolling(params.atr_bars, min_periods=params.atr_bars).mean()
        prior_atr = atr.shift(1).where(lambda s: s > 0)
        gap_atr = (prices["open"] - previous_close) / prior_atr
        valid = prices.notna().all(axis=1) & prices.gt(0).all(axis=1)
        rejected_down = (valid & (gap_atr < -params.min_gap_atr) & (close > previous_close)).fillna(False)
        rejected_up = (valid & (gap_atr > params.min_gap_atr) & (close < previous_close)).fillna(False)
        down_streak = rejected_down.astype(int).groupby((~rejected_down).cumsum()).cumsum()
        up_streak = rejected_up.astype(int).groupby((~rejected_up).cumsum()).cumsum()
        ma200 = close.rolling(200, min_periods=200).mean()
        ready = valid & ma200.notna() & atr.notna() & (atr > 0) & np.isfinite(gap_atr)
        entry = pd.Series(0, index=data.index, dtype=int)
        entry.loc[ready & (close > ma200) & (down_streak == params.streak_bars)] = 1
        entry.loc[ready & (close < ma200) & (up_streak == params.streak_bars)] = -1
        return pd.DataFrame({
            "previous_close": previous_close,
            "true_range": true_range,
            "atr": atr,
            "prior_atr": prior_atr,
            "gap_atr": gap_atr,
            "ma200": ma200,
            "rejected_down": rejected_down,
            "rejected_up": rejected_up,
            "down_streak": down_streak,
            "up_streak": up_streak,
            "ready": ready,
            "entry": entry,
        }, index=data.index)

    def generate_signals(self, data, indicators, ctx, params):
        closes = data["close"].to_numpy(dtype=float)
        atrs = indicators["atr"].to_numpy(dtype=float)
        entries = indicators["entry"].to_numpy(dtype=int)
        ready = indicators["ready"].to_numpy(dtype=bool)
        decisions = np.zeros(len(data), dtype=int)
        position = 0
        held = 0
        extreme = np.nan
        stop = np.nan
        for i in range(len(data)):
            if position:
                held += 1
                if not ready[i] or not np.isfinite(closes[i]) or not np.isfinite(atrs[i]):
                    position = 0
                else:
                    # Entry fills on this bar after yesterday's decision.
                    # Only closes observed during the actual holding period count.
                    if position == 1:
                        extreme = closes[i] if np.isnan(extreme) else max(extreme, closes[i])
                        candidate = extreme - params.trailing_atr * atrs[i]
                        stop = candidate if np.isnan(stop) else max(stop, candidate)
                        stopped = closes[i] <= stop
                    else:
                        # Mirror the close-high trailing stop for short positions.
                        extreme = closes[i] if np.isnan(extreme) else min(extreme, closes[i])
                        candidate = extreme + params.trailing_atr * atrs[i]
                        stop = candidate if np.isnan(stop) else min(stop, candidate)
                        stopped = closes[i] >= stop
                    if stopped or held >= params.max_hold_bars:
                        position = 0
            elif ready[i] and entries[i] != 0:
                position = int(entries[i])
                held = 0
                extreme = np.nan
                stop = np.nan
            decisions[i] = position
        df = pd.DataFrame({"signal": decisions, "size": 1.0}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check():
    strategy = GeneratedStrategy()
    index = pd.date_range("2020-01-01", periods=8, freq="D")
    params = GapRejectionParams(trailing_atr=2.0)
    for direction, closes in [(1, [100, 101, 105, 104, 103, 102, 101, 100]), (-1, [100, 99, 95, 96, 97, 98, 99, 100])]:
        data = pd.DataFrame({"close": closes}, index=index)
        indicators = pd.DataFrame({"atr": [1, 1, 1, 10, 10, 10, 10, 10], "entry": [direction, 0, 0, 0, 0, 0, 0, 0], "ready": True}, index=index)
        result = strategy.generate_signals(data, indicators, None, params).data
        assert result["signal"].tolist() == [0, direction, direction, direction, direction, 0, 0, 0]
        assert result["size"].gt(0).all()
    params.max_hold_bars = 2
    result = strategy.generate_signals(data, indicators, None, params).data
    assert result["signal"].tolist() == [0, -1, -1, 0, 0, 0, 0, 0]
    assert strategy.warmup_bars(params) >= 200


if __name__ == "__main__":
    _self_check()
