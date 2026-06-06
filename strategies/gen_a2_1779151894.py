from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class GeneratedParams:
    lookback: int = 60
    depth_floor: float = 0.04
    depth_cap: float = 0.15
    uw_eps: float = 0.005
    capacity: int = 15
    atr_window: int = 14
    init_stop_mult: float = 2.5
    breakeven_pct: float = 0.03
    trail_mult: float = 3.0
    max_hold_bars: int = 20


class GeneratedStrategy(BaseStrategy[GeneratedParams]):
    strategy_id = "gen_a2_1779151894"

    @classmethod
    def params_type(cls) -> type[GeneratedParams]:
        return GeneratedParams

    @staticmethod
    def warmup_bars(params: GeneratedParams) -> int:
        return int(max(params.lookback, params.atr_window) + 2)

    @staticmethod
    def indicators(data: pd.DataFrame, params: GeneratedParams) -> pd.DataFrame:
        close = data["close"]
        high = data["high"]
        low = data["low"]

        lookback = max(int(params.lookback), 2)
        atr_window = max(int(params.atr_window), 2)

        roll_max = close.rolling(lookback, min_periods=lookback).max()
        drawdown = close / roll_max - 1.0
        depth = (-drawdown).clip(lower=0.0)

        # Capacity-limited queue: count of consecutive underwater bars.
        underwater = (depth > float(params.uw_eps)).fillna(False).astype(int)
        reset_grp = (underwater == 0).cumsum()
        time_underwater = underwater.groupby(reset_grp).cumsum().astype(float)

        prev_close = close.shift(1)
        tr = pd.concat(
            [
                (high - low).abs(),
                (high - prev_close).abs(),
                (low - prev_close).abs(),
            ],
            axis=1,
        ).max(axis=1)
        atr = tr.rolling(atr_window, min_periods=atr_window).mean()

        depth_ok = (depth >= float(params.depth_floor)) & (depth <= float(params.depth_cap))
        queue_overflow = time_underwater >= float(params.capacity)
        entry = (depth_ok & queue_overflow & depth.notna() & atr.notna()).astype(float)

        out = pd.DataFrame(index=data.index)
        out["depth"] = depth
        out["tuw"] = time_underwater
        out["atr"] = atr
        out["entry"] = entry
        return out

    @staticmethod
    def generate_signals(
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: GeneratedParams,
    ) -> SignalFrame:
        close = data["close"].to_numpy(dtype=float)
        high = data["high"].to_numpy(dtype=float)
        low = data["low"].to_numpy(dtype=float)
        atr = indicators["atr"].to_numpy(dtype=float)
        entry = indicators["entry"].to_numpy(dtype=float)

        n = len(close)
        state = np.zeros(n, dtype=int)

        breakeven_pct = float(params.breakeven_pct)
        init_stop_mult = float(params.init_stop_mult)
        trail_mult = float(params.trail_mult)
        max_hold = int(params.max_hold_bars)

        position = 0
        entry_price = 0.0
        stop = 0.0
        bars_held = 0
        armed = False

        for i in range(n):
            if position == 0:
                a = atr[i]
                if entry[i] >= 1.0 and np.isfinite(a) and a > 0.0:
                    position = 1
                    entry_price = close[i]
                    stop = entry_price - init_stop_mult * a
                    bars_held = 0
                    armed = False
                    state[i] = 1
                else:
                    state[i] = 0
            else:
                bars_held += 1
                a = atr[i]
                if not np.isfinite(a) or a <= 0.0:
                    a = 0.0

                if not armed and high[i] >= entry_price * (1.0 + breakeven_pct):
                    armed = True
                    if entry_price > stop:
                        stop = entry_price

                if armed and a > 0.0:
                    trail = high[i] - trail_mult * a
                    if trail > stop:
                        stop = trail

                exit_now = (low[i] <= stop) or (bars_held >= max_hold)
                if exit_now:
                    position = 0
                    state[i] = 0
                else:
                    state[i] = 1

        df = pd.DataFrame(index=data.index)
        df["signal"] = state
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = 1.0
        return SignalFrame(data=df, signal_column="signal", size_column="size")
