from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class EscapeVelocityParams:
    channel_len: int = 20
    roc_len: int = 5
    accel_smooth: int = 3
    accel_vol_len: int = 30
    escape_mult: float = 1.0
    atr_len: int = 14
    trail_mult: float = 2.5
    breakeven_pct: float = 0.02
    max_hold: int = 5
    refractory: int = 3
    base_size: float = 0.5
    size_gain: float = 0.6
    max_size: float = 1.0


class GeneratedStrategy(BaseStrategy[EscapeVelocityParams]):
    strategy_id = "gen_a1_1778907625"

    @classmethod
    def params_type(cls):
        return EscapeVelocityParams

    def warmup_bars(self, params):
        accel_chain = params.roc_len + params.accel_smooth + params.accel_vol_len + 1
        return int(max(params.channel_len + 1, accel_chain, params.atr_len + 1)) + 5

    def indicators(self, data, params):
        high = data["high"]
        low = data["low"]
        close = data["close"]

        ind = pd.DataFrame(index=data.index)
        ind["donchian_high"] = high.rolling(params.channel_len).max().shift(1)
        ind["donchian_low"] = low.rolling(params.channel_len).min().shift(1)

        roc = close.pct_change(params.roc_len)
        roc_s = roc.rolling(params.accel_smooth).mean()
        accel = roc_s.diff()
        ind["accel"] = accel
        ind["escape"] = params.escape_mult * accel.rolling(params.accel_vol_len).std()

        prev_close = close.shift(1)
        tr = pd.concat(
            [(high - low), (high - prev_close).abs(), (low - prev_close).abs()],
            axis=1,
        ).max(axis=1)
        ind["atr"] = tr.rolling(params.atr_len).mean()
        return ind

    def generate_signals(self, data, indicators, ctx, params):
        close = data["close"].to_numpy(dtype=float)
        dh = indicators["donchian_high"].to_numpy(dtype=float)
        dl = indicators["donchian_low"].to_numpy(dtype=float)
        accel = indicators["accel"].to_numpy(dtype=float)
        escape = indicators["escape"].to_numpy(dtype=float)
        atr = indicators["atr"].to_numpy(dtype=float)

        n = len(close)
        sig = np.zeros(n, dtype=int)
        size = np.full(n, params.base_size, dtype=float)

        in_pos = False
        side = 0
        entry_price = 0.0
        entry_i = 0
        stop = 0.0
        armed = False
        cooldown = 0
        cur_size = params.base_size

        for i in range(n):
            c = close[i]
            a = accel[i]
            esc = escape[i]
            atr_i = atr[i]

            if cooldown > 0:
                cooldown -= 1

            if in_pos:
                held = i - entry_i
                exit_now = False
                if not np.isnan(atr_i) and atr_i > 0.0:
                    if side == 1:
                        if not armed and c >= entry_price * (1.0 + params.breakeven_pct):
                            armed = True
                            stop = max(stop, entry_price)
                        if armed:
                            stop = max(stop, c - params.trail_mult * atr_i)
                        if c <= stop:
                            exit_now = True
                    else:
                        if not armed and c <= entry_price * (1.0 - params.breakeven_pct):
                            armed = True
                            stop = min(stop, entry_price)
                        if armed:
                            stop = min(stop, c + params.trail_mult * atr_i)
                        if c >= stop:
                            exit_now = True
                if held >= params.max_hold:
                    exit_now = True

                if exit_now:
                    in_pos = False
                    side = 0
                    armed = False
                    cooldown = params.refractory
                    sig[i] = 0
                    size[i] = params.base_size
                else:
                    sig[i] = side
                    size[i] = cur_size
                continue

            valid = (
                not np.isnan(a)
                and not np.isnan(esc)
                and not np.isnan(dh[i])
                and not np.isnan(dl[i])
                and not np.isnan(atr_i)
                and esc > 0.0
                and atr_i > 0.0
            )
            if valid and cooldown == 0:
                long_cand = c > dh[i] and a > esc
                short_cand = c < dl[i] and a < -esc
                if long_cand:
                    excess = (a - esc) / esc
                    cur_size = float(
                        np.clip(
                            params.base_size + params.size_gain * excess,
                            params.base_size,
                            params.max_size,
                        )
                    )
                    in_pos = True
                    side = 1
                    entry_price = c
                    entry_i = i
                    armed = False
                    stop = c - params.trail_mult * atr_i
                    sig[i] = 1
                    size[i] = cur_size
                elif short_cand:
                    excess = (-a - esc) / esc
                    cur_size = float(
                        np.clip(
                            params.base_size + params.size_gain * excess,
                            params.base_size,
                            params.max_size,
                        )
                    )
                    in_pos = True
                    side = -1
                    entry_price = c
                    entry_i = i
                    armed = False
                    stop = c + params.trail_mult * atr_i
                    sig[i] = -1
                    size[i] = cur_size

        df = pd.DataFrame(index=data.index)
        df["signal"] = sig
        df["size"] = size
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = df["size"].shift(1).fillna(params.base_size)
        return SignalFrame(data=df, signal_column="signal", size_column="size")
