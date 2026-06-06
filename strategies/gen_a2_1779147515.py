from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class Params:
    atr_len: int = 14
    breakout_len: int = 20
    spark_k: float = 0.5
    epi_window: int = 10
    profit_target: float = 0.02
    time_stop: int = 2
    base_size: float = 1.0


class GeneratedStrategy(BaseStrategy[Params]):
    strategy_id = "gen_a2_1779147515"

    @classmethod
    def params_type(cls):
        return Params

    def warmup_bars(self, params: Params) -> int:
        return int(max(params.atr_len, params.breakout_len, params.epi_window)) + 2

    def indicators(self, data: pd.DataFrame, params: Params) -> pd.DataFrame:
        high = data["high"]
        low = data["low"]
        close = data["close"]
        prev_close = close.shift(1)

        tr = pd.concat(
            [
                (high - low),
                (high - prev_close).abs(),
                (low - prev_close).abs(),
            ],
            axis=1,
        ).max(axis=1)
        atr = tr.rolling(params.atr_len, min_periods=1).mean()

        # prior breakout reference high (shifted to avoid using the current bar)
        roll_high = high.rolling(params.breakout_len, min_periods=1).max().shift(1)

        # a "spark": close clears the prior breakout high by a volatility-scaled margin
        spark = (close > (roll_high + params.spark_k * atr)).fillna(False)

        # epidemic curve: how many of the recent bars are infected (sparked)
        infection = spark.rolling(params.epi_window, min_periods=1).sum()
        susceptible = (1.0 - infection / float(params.epi_window)).clip(0.0, 1.0)

        out = pd.DataFrame(index=data.index)
        out["atr"] = atr.fillna(0.0)
        out["spark"] = spark.astype(float)
        out["infection"] = infection.fillna(0.0)
        out["susceptible"] = susceptible.fillna(1.0)
        return out

    def generate_signals(self, data, indicators, ctx, params) -> SignalFrame:
        close = data["close"].to_numpy(dtype=float)
        spark = indicators["spark"].to_numpy(dtype=float) > 0.5
        susceptible = indicators["susceptible"].to_numpy(dtype=float)
        n = len(close)

        # two-bar confirmation: entry requires two consecutive spark bars
        confirm = np.zeros(n, dtype=bool)
        if n > 1:
            confirm[1:] = spark[1:] & spark[:-1]

        position = np.zeros(n, dtype=int)
        in_pos = False
        entry_price = 0.0
        bars_in_trade = 0
        pt = float(params.profit_target)
        ts = int(params.time_stop)

        for i in range(n):
            if not in_pos:
                if confirm[i]:
                    in_pos = True
                    entry_price = close[i]
                    bars_in_trade = 0
                    position[i] = 1
                else:
                    position[i] = 0
            else:
                bars_in_trade += 1
                hit_target = entry_price > 0.0 and close[i] >= entry_price * (1.0 + pt)
                hit_time = bars_in_trade >= ts
                if hit_target or hit_time:
                    in_pos = False
                    entry_price = 0.0
                    bars_in_trade = 0
                    position[i] = 0
                else:
                    position[i] = 1

        df = pd.DataFrame(index=data.index)
        df["signal"] = position
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)

        # size by the susceptible pool: a fresh epidemic gets a larger position
        susc = np.nan_to_num(susceptible, nan=1.0)
        size = float(params.base_size) * (0.5 + susc)
        size = np.clip(size, 0.1, None)
        df["size"] = size

        return SignalFrame(data=df, signal_column="signal", size_column="size")
