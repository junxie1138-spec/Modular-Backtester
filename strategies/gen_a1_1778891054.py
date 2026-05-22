from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class GeneratedParams:
    ma_len: int = 50
    z_len: int = 40
    entry_z: float = 1.0
    exit_z: float = 1.25
    confirm_bars: int = 2


class GeneratedStrategy(BaseStrategy[GeneratedParams]):
    strategy_id = "gen_a1_1778891054"

    @classmethod
    def params_type(cls):
        return GeneratedParams

    @staticmethod
    def warmup_bars(params: GeneratedParams) -> int:
        return int(params.ma_len + params.z_len + 5)

    def indicators(self, data: pd.DataFrame, params: GeneratedParams) -> pd.DataFrame:
        close = data["close"]
        ma = close.rolling(params.ma_len, min_periods=params.ma_len).mean()
        dist = close - ma
        sd = dist.rolling(params.z_len, min_periods=params.z_len).std()
        sd = sd.replace(0.0, np.nan)
        z = dist / sd

        out = pd.DataFrame(index=data.index)
        out["ma"] = ma
        out["dist"] = dist
        out["z"] = z
        return out

    def generate_signals(self, data, indicators, ctx, params):
        z = indicators["z"]
        n = len(z)
        zv = z.to_numpy(dtype=float)

        valid = ~np.isnan(zv)
        zfilled = np.where(valid, zv, 0.0)

        entry_z = float(params.entry_z)
        exit_z = float(params.exit_z)
        confirm = max(2, int(params.confirm_bars))

        # 'escaped' = z above the lower discount band; 'jam' = z inside discount.
        escaped = valid & (zfilled > -entry_z)
        jam = valid & (zfilled <= -entry_z)

        raw = np.zeros(n, dtype=int)
        in_pos = False

        for i in range(n):
            if not in_pos:
                if i < confirm:
                    continue
                # two-bar (confirm-bar) confirmation: escaped for the whole
                # confirmation run, the bar just before it was in the jam,
                # and the z-score rose across the run (back edge advancing).
                window_escaped = escaped[i - confirm + 1:i + 1].all()
                jam_before = jam[i - confirm]
                rising = zfilled[i] > zfilled[i - confirm + 1]
                if window_escaped and jam_before and rising:
                    in_pos = True
                    raw[i] = 1
            else:
                # signal-reversal exit: mirror flip into the premium band.
                if valid[i] and zv[i] >= exit_z:
                    in_pos = False
                    raw[i] = 0
                else:
                    raw[i] = 1

        df = pd.DataFrame(index=data.index)
        df["signal"] = raw
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = 1.0
        return SignalFrame(data=df, signal_column="signal", size_column="size")
