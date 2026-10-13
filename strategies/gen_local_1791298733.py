from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class Params:
    streak_len: int = 3
    breakout_len: int = 12
    trail_atr: float = 1.2
    trigger_pct: float = 0.005
    max_hold: int = 2


class GeneratedStrategy(BaseStrategy[Params]):
    strategy_id = "gen_local_1791298733"

    @classmethod
    def params_type(cls) -> type[Params]:
        return Params

    @classmethod
    def warmup_bars(cls, params: Params) -> int:
        return max(200, params.breakout_len + 1, 15)

    def indicators(self, data: pd.DataFrame, params: Params) -> pd.DataFrame:
        out = pd.DataFrame(index=data.index)
        close = data["close"]
        up = close.gt(close.shift(1))
        down = close.lt(close.shift(1))
        out["up_streak"] = up.groupby((~up).cumsum()).cumsum().astype(int)
        out["down_streak"] = down.groupby((~down).cumsum()).cumsum().astype(int)
        out["upper"] = data["high"].shift(1).rolling(params.breakout_len, min_periods=params.breakout_len).max().fillna(0.0)
        out["lower"] = data["low"].shift(1).rolling(params.breakout_len, min_periods=params.breakout_len).min().fillna(0.0)
        out["ma200"] = close.rolling(200, min_periods=200).mean().fillna(0.0)
        previous_close = close.shift(1)
        true_range = pd.concat((data["high"] - data["low"], (data["high"] - previous_close).abs(), (data["low"] - previous_close).abs()), axis=1).max(axis=1)
        out["atr"] = true_range.rolling(14, min_periods=14).mean().fillna(0.0)
        return out

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: Params) -> SignalFrame:
        close = data["close"].to_numpy(dtype=float)
        opened = data["open"].to_numpy(dtype=float)
        atr = indicators["atr"].to_numpy(dtype=float)
        long_entry = ((indicators["up_streak"] >= params.streak_len) & (data["close"] > indicators["upper"]) & (data["close"] > indicators["ma200"])).to_numpy(dtype=bool)
        short_entry = ((indicators["down_streak"] >= params.streak_len) & (data["close"] < indicators["lower"]) & (data["close"] < indicators["ma200"])).to_numpy(dtype=bool)
        raw = np.zeros(len(data), dtype=int)
        state = 0
        bars = 0
        entry = 0.0
        stop = 0.0
        armed = False

        for i in range(len(data)):
            exited = False
            if state:
                if bars == 0:
                    entry = opened[i]
                    stop = entry - state * params.trail_atr * atr[i]
                    armed = False
                bars += 1
                if (state == 1 and close[i] <= stop) or (state == -1 and close[i] >= stop) or bars >= params.max_hold:
                    state = 0
                    bars = 0
                    exited = True
                elif state == 1:
                    armed = armed or close[i] >= entry * (1.0 + params.trigger_pct)
                    if armed:
                        stop = max(stop, entry, close[i] - params.trail_atr * atr[i])
                else:
                    armed = armed or close[i] <= entry * (1.0 - params.trigger_pct)
                    if armed:
                        stop = min(stop, entry, close[i] + params.trail_atr * atr[i])

            if state == 0 and not exited and i >= self.warmup_bars(params) and atr[i] > 0:
                if long_entry[i]:
                    state = 1
                elif short_entry[i]:
                    state = -1
            raw[i] = state

        df = pd.DataFrame(index=data.index)
        df["signal"] = pd.Series(raw, index=data.index).shift(1).fillna(0).astype(int)
        df["size"] = 1.0
        return SignalFrame(data=df, signal_column="signal", size_column="size")
