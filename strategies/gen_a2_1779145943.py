from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class Params:
    rp_window: int = 20
    ma_window: int = 200
    low_zone: float = 0.20
    high_zone: float = 0.80
    streak_len: int = 3
    size_gain: float = 0.10
    size_cap: float = 5.0


class GeneratedStrategy(BaseStrategy[Params]):
    strategy_id = "gen_a2_1779145943"

    @classmethod
    def params_type(cls) -> type[Params]:
        return Params

    @staticmethod
    def warmup_bars(params: Params) -> int:
        return int(max(params.ma_window, params.rp_window) + params.streak_len + 2)

    @staticmethod
    def indicators(data: pd.DataFrame, params: Params) -> pd.DataFrame:
        close = data["close"]
        high = data["high"]
        low = data["low"]

        rp_w = max(int(params.rp_window), 2)
        roll_min = low.rolling(rp_w, min_periods=rp_w).min()
        roll_max = high.rolling(rp_w, min_periods=rp_w).max()
        span = (roll_max - roll_min).replace(0.0, np.nan)
        rp = ((close - roll_min) / span).clip(lower=0.0, upper=1.0)

        ma_w = max(int(params.ma_window), 2)
        ma = close.rolling(ma_w, min_periods=ma_w).mean()

        floor_cond = (rp <= params.low_zone).fillna(False)
        ceil_cond = (rp >= params.high_zone).fillna(False)

        floor_streak = floor_cond.groupby((~floor_cond).cumsum()).cumsum()
        ceil_streak = ceil_cond.groupby((~ceil_cond).cumsum()).cumsum()

        out = pd.DataFrame(index=data.index)
        out["rp"] = rp
        out["ma"] = ma
        out["floor_streak"] = floor_streak.astype(float)
        out["ceil_streak"] = ceil_streak.astype(float)
        return out

    @staticmethod
    def generate_signals(
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: Params,
    ) -> SignalFrame:
        close = data["close"].to_numpy(dtype=float)
        ma = indicators["ma"].to_numpy(dtype=float)
        floor_streak = indicators["floor_streak"].to_numpy(dtype=float)
        ceil_streak = indicators["ceil_streak"].to_numpy(dtype=float)

        n = len(close)
        signal = np.zeros(n, dtype=int)
        size = np.ones(n, dtype=float)

        sl = max(int(params.streak_len), 1)
        gain = float(params.size_gain)
        cap = float(params.size_cap)
        pos = 0

        for i in range(n):
            m = ma[i]
            fs = floor_streak[i]
            cs = ceil_streak[i]

            if not np.isfinite(m) or not np.isfinite(fs) or not np.isfinite(cs):
                pos = 0
                signal[i] = 0
                size[i] = 1.0
                continue

            up = close[i] > m
            down = close[i] < m

            # Signal-reversal exit: leave when the entry condition flips,
            # i.e. price escapes the streak zone or the regime turns.
            if pos == 1 and (fs <= 0.0 or down):
                pos = 0
            elif pos == -1 and (cs <= 0.0 or up):
                pos = 0

            # Entry only when a compression streak fires with the regime.
            if pos == 0:
                if up and fs >= sl:
                    pos = 1
                elif down and cs >= sl:
                    pos = -1

            signal[i] = pos
            if pos == 1:
                depth = min(max(fs - sl, 0.0), cap)
                size[i] = 1.0 + gain * depth
            elif pos == -1:
                depth = min(max(cs - sl, 0.0), cap)
                size[i] = 1.0 + gain * depth
            else:
                size[i] = 1.0

        df = pd.DataFrame(index=data.index)
        df["signal"] = signal
        df["size"] = size
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = df["size"].clip(lower=0.01)
        return SignalFrame(data=df, signal_column="signal", size_column="size")
