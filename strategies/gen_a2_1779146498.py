from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class Params:
    ma_len: int = 20
    z_len: int = 20
    entry_z: float = 1.5
    vol_len: int = 20
    vol_target: float = 0.15
    max_leverage: float = 2.0
    min_size: float = 0.10


class GeneratedStrategy(BaseStrategy[Params]):
    strategy_id = "gen_a2_1779146498"

    @classmethod
    def params_type(cls):
        return Params

    @staticmethod
    def warmup_bars(params: Params) -> int:
        return int(params.ma_len + params.z_len + params.vol_len + 5)

    @staticmethod
    def indicators(data: pd.DataFrame, params: Params) -> pd.DataFrame:
        close = data["close"]
        ind = pd.DataFrame(index=data.index)

        # Distance-from-MA z-score: how stretched price is vs its own mean.
        ma = close.rolling(params.ma_len).mean()
        dist = close - ma
        dist_mean = dist.rolling(params.z_len).mean()
        dist_std = dist.rolling(params.z_len).std()
        z = (dist - dist_mean) / dist_std.replace(0.0, np.nan)
        ind["z"] = z

        # Volatility-targeted sizing: scale exposure inversely to realized vol.
        ret = close.pct_change()
        rvol = ret.rolling(params.vol_len).std() * np.sqrt(252.0)
        raw = params.vol_target / rvol.replace(0.0, np.nan)
        size = raw.clip(lower=params.min_size, upper=params.max_leverage)
        ind["size"] = size.fillna(params.min_size)
        return ind

    @staticmethod
    def generate_signals(data, indicators, ctx, params) -> SignalFrame:
        df = pd.DataFrame(index=data.index)
        z = indicators["z"]

        # Raw extreme conditions (NaN comparisons resolve to False).
        long_cond = (z < -params.entry_z).fillna(False).to_numpy()
        short_cond = (z > params.entry_z).fillna(False).to_numpy()

        n = len(df)
        long_fire = np.zeros(n, dtype=bool)
        short_fire = np.zeros(n, dtype=bool)
        # Two-bar confirmation: the extreme must persist across two closes.
        for i in range(1, n):
            long_fire[i] = bool(long_cond[i] and long_cond[i - 1])
            short_fire[i] = bool(short_cond[i] and short_cond[i - 1])

        # Signal-reversal exit: a position is held until the OPPOSITE confirmed
        # entry fires - the exit is exactly the entry condition flipping.
        sig = np.zeros(n, dtype=int)
        pos = 0
        for i in range(n):
            if pos == 0:
                if long_fire[i]:
                    pos = 1
                elif short_fire[i]:
                    pos = -1
            elif pos == 1:
                if short_fire[i]:
                    pos = -1
            elif pos == -1:
                if long_fire[i]:
                    pos = 1
            sig[i] = pos

        df["signal"] = sig
        # MANDATORY one-bar shift: decide on bar N close, fill on bar N+1.
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)

        size = indicators["size"].astype(float)
        size = size.where(size > 0.0, params.min_size).fillna(params.min_size)
        df["size"] = size

        return SignalFrame(data=df, signal_column="signal", size_column="size")
