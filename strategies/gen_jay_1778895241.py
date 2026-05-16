from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class Params:
    fast_window: int = 10
    slow_window: int = 50
    z_fast_entry: float = -2.0
    z_slow_floor: float = -0.5
    atr_window: int = 14
    trail_k: float = 2.5
    size_cap: float = 0.95


class GeneratedStrategy(BaseStrategy[Params]):
    strategy_id = "gen_jay_1778895241"

    @classmethod
    def params_type(cls) -> type[Params]:
        return Params

    @staticmethod
    def warmup_bars(params: Params) -> int:
        return params.slow_window + params.atr_window + 2

    @staticmethod
    def indicators(data: pd.DataFrame, params: Params) -> pd.DataFrame:
        close = data["close"]
        high = data["high"]
        low = data["low"]

        ind = pd.DataFrame(index=data.index)

        fast_ma = close.rolling(params.fast_window).mean()
        fast_std = close.rolling(params.fast_window).std(ddof=1)
        ind["z_fast"] = (close - fast_ma) / fast_std.where(fast_std > 0)

        slow_ma = close.rolling(params.slow_window).mean()
        slow_std = close.rolling(params.slow_window).std(ddof=1)
        ind["z_slow"] = (close - slow_ma) / slow_std.where(slow_std > 0)

        tr = pd.concat(
            [
                high - low,
                (high - close.shift(1)).abs(),
                (low - close.shift(1)).abs(),
            ],
            axis=1,
        ).max(axis=1)
        ind["atr"] = tr.rolling(params.atr_window).mean()

        return ind

    @staticmethod
    def generate_signals(
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: Params,
    ) -> SignalFrame:
        close = data["close"].values
        z_fast = indicators["z_fast"].values
        z_slow = indicators["z_slow"].values
        atr = indicators["atr"].values

        n = len(close)
        raw_signal = np.zeros(n, dtype=int)
        raw_size = np.full(n, 0.1)

        in_trade = False
        high_water = 0.0

        for i in range(1, n):
            if (
                np.isnan(z_fast[i])
                or np.isnan(z_slow[i])
                or np.isnan(atr[i])
                or atr[i] <= 0.0
            ):
                raw_signal[i] = 0
                raw_size[i] = 0.1
                in_trade = False
                continue

            if in_trade:
                if close[i] > high_water:
                    high_water = close[i]

                stop_level = high_water - params.trail_k * atr[i]
                if close[i] < stop_level:
                    raw_signal[i] = 0
                    in_trade = False
                else:
                    raw_signal[i] = 1
                    z_mag = min(abs(z_fast[i]), 3.0) / 3.0
                    raw_size[i] = min(0.1 + 0.85 * z_mag, params.size_cap)
            else:
                if z_fast[i] <= params.z_fast_entry and z_slow[i] >= params.z_slow_floor:
                    raw_signal[i] = 1
                    in_trade = True
                    high_water = close[i]
                    z_mag = min(abs(z_fast[i]), 3.0) / 3.0
                    raw_size[i] = min(0.1 + 0.85 * z_mag, params.size_cap)
                else:
                    raw_signal[i] = 0
                    raw_size[i] = 0.1

        df = data[["close"]].copy()
        df["signal"] = raw_signal
        df["size"] = raw_size
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = df["size"].shift(1).fillna(0.1)

        return SignalFrame(data=df, signal_column="signal", size_column="size")
