from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class Params:
    streak_len: int = 3
    strength_thresh: float = 0.65
    vol_ema_fast: int = 5
    vol_ema_slow: int = 20
    atr_period: int = 14
    atr_stop_mult: float = 2.0
    refractory_bars: int = 5
    vol_spike_mult: float = 2.5


class GeneratedStrategy(BaseStrategy[Params]):
    strategy_id = "gen_jay_1778912050"

    @classmethod
    def params_type(cls):
        return Params

    @staticmethod
    def warmup_bars(params: Params) -> int:
        return params.vol_ema_slow + params.atr_period + params.streak_len + 5

    @staticmethod
    def indicators(data: pd.DataFrame, params: Params) -> pd.DataFrame:
        ind = pd.DataFrame(index=data.index)

        # Fraction of day's range where close lands (intraday strength)
        rng = (data["high"] - data["low"]).replace(0, np.nan)
        ind["intraday_strength"] = ((data["close"] - data["low"]) / rng).fillna(0.5)

        # Consecutive bars where close lands in upper portion of day's range
        strong = (ind["intraday_strength"] >= params.strength_thresh).astype(int)
        group = (strong == 0).cumsum()
        ind["streak"] = strong.groupby(group).cumsum()

        # Volume acceleration: fast EMA / slow EMA ratio
        vol_ema_fast = data["volume"].ewm(span=params.vol_ema_fast, adjust=False).mean()
        vol_ema_slow = data["volume"].ewm(span=params.vol_ema_slow, adjust=False).mean()
        ind["vol_acceleration"] = vol_ema_fast / vol_ema_slow.replace(0, np.nan)

        # Volume spike flag: single-bar volume exceeds rolling mean by spike_mult
        vol_roll_mean = data["volume"].rolling(params.vol_ema_slow).mean()
        ind["vol_spike"] = (data["volume"] > params.vol_spike_mult * vol_roll_mean).astype(int)

        # ATR for fixed volatility stop computation
        hl = data["high"] - data["low"]
        hpc = (data["high"] - data["close"].shift(1)).abs()
        lpc = (data["low"] - data["close"].shift(1)).abs()
        tr = pd.concat([hl, hpc, lpc], axis=1).max(axis=1)
        ind["atr"] = tr.rolling(params.atr_period).mean()

        return ind

    @staticmethod
    def generate_signals(
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: Params,
    ) -> SignalFrame:
        n = len(data)
        signal = np.zeros(n, dtype=int)
        size = np.ones(n, dtype=float)

        streak = indicators["streak"].values
        vol_acc = indicators["vol_acceleration"].values
        vol_spike = indicators["vol_spike"].values
        atr = indicators["atr"].values
        close = data["close"].values

        in_position = False
        entry_price = 0.0
        entry_atr = 0.0
        refractory_counter = 0

        for i in range(n):
            # Volume spike resets refractory cooldown; otherwise decrement
            if vol_spike[i]:
                refractory_counter = params.refractory_bars
            elif refractory_counter > 0:
                refractory_counter -= 1

            if in_position:
                # Fixed volatility stop: anchor is entry_atr, never updated
                stop_price = entry_price - params.atr_stop_mult * entry_atr
                if close[i] <= stop_price:
                    in_position = False
                    signal[i] = 0
                else:
                    signal[i] = 1
            else:
                atr_valid = not np.isnan(atr[i])
                vol_valid = not np.isnan(vol_acc[i])
                if (
                    atr_valid
                    and vol_valid
                    and streak[i] >= params.streak_len
                    and vol_acc[i] > 1.0
                    and refractory_counter == 0
                ):
                    in_position = True
                    entry_price = close[i]
                    entry_atr = atr[i]
                    signal[i] = 1

        df = pd.DataFrame({"signal": signal, "size": size}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)

        return SignalFrame(data=df, signal_column="signal", size_column="size")
