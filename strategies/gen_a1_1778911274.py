from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class GenParams:
    ma_len: int = 20
    z_len: int = 60
    atr_len: int = 14
    entry_z: float = 1.0
    spike_z: float = 2.5
    refractory_bars: int = 5
    atr_mult: float = 2.5
    size_max: float = 2.5
    allow_short: bool = True


class GeneratedStrategy(BaseStrategy[GenParams]):
    strategy_id = "gen_a1_1778911274"

    @classmethod
    def params_type(cls):
        return GenParams

    @staticmethod
    def warmup_bars(params: GenParams) -> int:
        # distance-from-MA needs ma_len bars, then a z_len rolling window over it.
        ma_z = params.ma_len + params.z_len
        atr_need = params.atr_len + 1  # true range uses the previous close
        return int(max(ma_z, atr_need)) + 5

    @staticmethod
    def indicators(data: pd.DataFrame, params: GenParams) -> pd.DataFrame:
        close = data["close"].astype(float)
        high = data["high"].astype(float)
        low = data["low"].astype(float)

        ma = close.rolling(params.ma_len, min_periods=params.ma_len).mean()
        dist = close - ma

        d_mean = dist.rolling(params.z_len, min_periods=params.z_len).mean()
        d_std = dist.rolling(params.z_len, min_periods=params.z_len).std(ddof=0)
        # avoid divide-by-zero: a flat distance series yields NaN z (treated as invalid)
        dist_z = (dist - d_mean) / d_std.replace(0.0, np.nan)

        prev_close = close.shift(1)
        tr = pd.concat(
            [
                (high - low),
                (high - prev_close).abs(),
                (low - prev_close).abs(),
            ],
            axis=1,
        ).max(axis=1)
        atr = tr.rolling(params.atr_len, min_periods=params.atr_len).mean()

        out = pd.DataFrame(index=data.index)
        out["dist_z"] = dist_z
        out["atr"] = atr
        return out

    @staticmethod
    def generate_signals(
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: GenParams,
    ) -> SignalFrame:
        close = data["close"].to_numpy(dtype=float)
        z = indicators["dist_z"].to_numpy(dtype=float)
        atr = indicators["atr"].to_numpy(dtype=float)
        n = len(close)

        raw_signal = np.zeros(n, dtype=int)
        raw_size = np.ones(n, dtype=float)

        entry_z = float(params.entry_z)
        spike_z = float(params.spike_z)
        k = float(params.atr_mult)
        refr = int(params.refractory_bars)
        size_max = float(params.size_max)
        allow_short = bool(params.allow_short)
        if entry_z <= 0.0:
            entry_z = 1e-9

        pos = 0
        hwm = 0.0          # highest close since a long entry (ratchets up)
        lwm = 0.0          # lowest close since a short entry (ratchets down)
        entry_z_mag = 0.0  # |z| captured at entry, drives signal-scaled size
        refractory_until = -1

        for i in range(n):
            zi = z[i]
            ai = atr[i]
            ci = close[i]
            valid = not (np.isnan(zi) or np.isnan(ai))

            # --- manage open position: rolling-high ATR trailing stop ---
            if pos == 1:
                if ci > hwm:
                    hwm = ci
                if valid and ci <= hwm - k * ai:
                    pos = 0
            elif pos == -1:
                if ci < lwm:
                    lwm = ci
                if valid and ci >= lwm + k * ai:
                    pos = 0

            # --- spike detection arms the refractory entry lockout ---
            if valid and abs(zi) >= spike_z:
                refractory_until = i + refr

            # --- entry only when flat and outside the refractory window ---
            if pos == 0 and valid and i > refractory_until:
                if zi >= entry_z:
                    pos = 1
                    hwm = ci
                    entry_z_mag = abs(zi)
                elif allow_short and zi <= -entry_z:
                    pos = -1
                    lwm = ci
                    entry_z_mag = abs(zi)

            raw_signal[i] = pos
            if pos != 0:
                s = entry_z_mag / entry_z
                if s < 1.0:
                    s = 1.0
                elif s > size_max:
                    s = size_max
                raw_size[i] = s
            else:
                raw_size[i] = 1.0

        df = pd.DataFrame(index=data.index)
        df["signal"] = raw_signal
        df["size"] = raw_size

        # MANDATORY one-bar shift: decide on bar N close, fill on bar N+1
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = df["size"].shift(1).fillna(1.0).astype(float)

        return SignalFrame(data=df, signal_column="signal", size_column="size")
