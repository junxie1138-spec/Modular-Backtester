from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class Params:
    vol_window: int = 20
    vol_mult: float = 1.2
    window: int = 15
    plastic_thresh: float = 0.40
    elastic_thresh: float = 0.15
    atr_window: int = 14
    atr_stop_k: float = 2.5
    max_hold: int = 5
    base_size: float = 1.0
    size_floor: float = 0.25


class GeneratedStrategy(BaseStrategy):
    strategy_id = "gen_a1_1779647614"

    @classmethod
    def params_type(cls):
        return Params

    @classmethod
    def warmup_bars(cls, params: Params) -> int:
        return int(params.vol_window + params.window + params.atr_window + 5)

    def indicators(self, data: pd.DataFrame, params: Params) -> pd.DataFrame:
        close = data["close"]
        high = data["high"]
        low = data["low"]
        volume = data["volume"]

        ret = close.pct_change()
        vol_sma = volume.rolling(params.vol_window, min_periods=params.vol_window).mean()
        is_confirmed = (volume > params.vol_mult * vol_sma).astype(float).fillna(0.0)
        signed_confirm = np.sign(ret).fillna(0.0) * is_confirmed
        vcms = signed_confirm.rolling(params.window, min_periods=params.window).sum()
        pressure = vcms / float(params.window)

        prev_close = close.shift(1)
        tr1 = high - low
        tr2 = (high - prev_close).abs()
        tr3 = (low - prev_close).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(params.atr_window, min_periods=params.atr_window).mean()

        return pd.DataFrame({
            "pressure": pressure,
            "atr": atr,
        }, index=data.index)

    def generate_signals(self, data, indicators, ctx, params):
        close_arr = data["close"].to_numpy(dtype=float)
        pressure = indicators["pressure"].to_numpy(dtype=float)
        atr = indicators["atr"].to_numpy(dtype=float)

        n = len(close_arr)
        raw_signal = np.zeros(n, dtype=np.int64)
        raw_size = np.ones(n, dtype=float)

        in_trade = 0
        entry_price = 0.0
        entry_atr = 0.0
        bars_held = 0
        active_size = 1.0

        span = params.plastic_thresh - params.elastic_thresh
        if span < 1e-9:
            span = 1e-9

        for i in range(n):
            p = pressure[i]
            a = atr[i]
            c = close_arr[i]
            p_ok = np.isfinite(p)
            valid = p_ok and np.isfinite(a) and a > 0.0 and np.isfinite(c)

            if in_trade != 0:
                exit_now = False
                if valid:
                    if in_trade == 1 and c < entry_price - params.atr_stop_k * entry_atr:
                        exit_now = True
                    elif in_trade == -1 and c > entry_price + params.atr_stop_k * entry_atr:
                        exit_now = True
                bars_held += 1
                if bars_held >= params.max_hold:
                    exit_now = True
                if exit_now:
                    raw_signal[i] = 0
                    raw_size[i] = active_size
                    in_trade = 0
                    entry_price = 0.0
                    entry_atr = 0.0
                    bars_held = 0
                    active_size = 1.0
                else:
                    raw_signal[i] = in_trade
                    raw_size[i] = active_size
            else:
                ap = abs(p) if p_ok else 0.0
                if valid and ap >= params.plastic_thresh:
                    strength = (ap - params.elastic_thresh) / span
                    if strength < 0.0:
                        strength = 0.0
                    elif strength > 1.0:
                        strength = 1.0
                    size_val = params.size_floor + (params.base_size - params.size_floor) * strength
                    if size_val <= 0.0:
                        size_val = params.size_floor if params.size_floor > 0.0 else 0.1
                    direction = 1 if p > 0 else -1
                    in_trade = direction
                    entry_price = c
                    entry_atr = a
                    bars_held = 0
                    active_size = size_val
                    raw_signal[i] = direction
                    raw_size[i] = size_val
                else:
                    raw_signal[i] = 0
                    raw_size[i] = 1.0

        df = pd.DataFrame({
            "signal": raw_signal,
            "size": raw_size,
        }, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = df["size"].shift(1).fillna(1.0)
        df.loc[df["size"] <= 0.0, "size"] = 1.0
        return SignalFrame(data=df, signal_column="signal", size_column="size")
