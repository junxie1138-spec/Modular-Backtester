from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class GeneratedParams:
    vol_window: int = 20
    vol_z_threshold: float = 1.5
    thrust_atr_mult: float = 0.5
    atr_window: int = 14
    atr_stop_mult: float = 2.5
    max_hold_bars: int = 5
    ma_window: int = 200
    regime_band: float = 0.01
    rvol_window: int = 20
    target_vol: float = 0.012
    size_floor: float = 0.3
    size_cap: float = 1.8


class GeneratedStrategy(BaseStrategy[GeneratedParams]):
    strategy_id = "gen_a1_1778910563"

    @classmethod
    def params_type(cls):
        return GeneratedParams

    def warmup_bars(self, params: GeneratedParams) -> int:
        return int(max(params.ma_window, params.vol_window,
                       params.atr_window, params.rvol_window)) + 2

    def indicators(self, data: pd.DataFrame, params: GeneratedParams) -> pd.DataFrame:
        close = data["close"].astype(float)
        high = data["high"].astype(float)
        low = data["low"].astype(float)
        volume = data["volume"].astype(float)

        ind = pd.DataFrame(index=data.index)

        ret = close.pct_change()
        ind["ret"] = ret

        prev_close = close.shift(1)
        tr = pd.concat([
            (high - low),
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ], axis=1).max(axis=1)
        atr = tr.rolling(params.atr_window, min_periods=params.atr_window).mean()
        ind["atr"] = atr
        ind["atr_pct"] = atr / close.replace(0.0, np.nan)

        vmean = volume.rolling(params.vol_window, min_periods=params.vol_window).mean()
        vstd = volume.rolling(params.vol_window, min_periods=params.vol_window).std()
        ind["vol_z"] = (volume - vmean) / vstd.replace(0.0, np.nan)

        ma = close.rolling(params.ma_window, min_periods=params.ma_window).mean()
        band = float(params.regime_band)
        ind["bull"] = (close > ma * (1.0 + band)).astype(float)
        ind["bear"] = (close < ma * (1.0 - band)).astype(float)

        rvol = ret.rolling(params.rvol_window, min_periods=params.rvol_window).std()
        ind["rvol"] = rvol

        return ind

    def generate_signals(self, data, indicators, ctx, params):
        close = data["close"].to_numpy(dtype=float)
        n = len(close)

        atr = indicators["atr"].to_numpy(dtype=float)
        atr_pct = indicators["atr_pct"].to_numpy(dtype=float)
        vol_z = indicators["vol_z"].to_numpy(dtype=float)
        ret = indicators["ret"].to_numpy(dtype=float)
        bull = indicators["bull"].to_numpy(dtype=float)
        bear = indicators["bear"].to_numpy(dtype=float)
        rvol = indicators["rvol"].to_numpy(dtype=float)

        sig = np.zeros(n, dtype=int)

        k = float(params.atr_stop_mult)
        z_thr = float(params.vol_z_threshold)
        thrust = float(params.thrust_atr_mult)
        max_hold = int(params.max_hold_bars)

        pos = 0
        bars_held = 0
        hwm = 0.0
        lwm = 0.0

        for i in range(n):
            ready = (np.isfinite(atr[i]) and np.isfinite(atr_pct[i])
                     and np.isfinite(vol_z[i]) and np.isfinite(ret[i]))
            if pos == 0:
                if not ready:
                    sig[i] = 0
                    continue
                move = thrust * atr_pct[i]
                long_ok = (vol_z[i] > z_thr and ret[i] > move and bull[i] > 0.5)
                short_ok = (vol_z[i] > z_thr and ret[i] < -move and bear[i] > 0.5)
                if long_ok:
                    pos = 1
                    bars_held = 0
                    hwm = close[i]
                    sig[i] = 1
                elif short_ok:
                    pos = -1
                    bars_held = 0
                    lwm = close[i]
                    sig[i] = -1
                else:
                    sig[i] = 0
            else:
                bars_held += 1
                exit_now = False
                atr_i = atr[i] if np.isfinite(atr[i]) else 0.0
                if pos == 1:
                    if close[i] > hwm:
                        hwm = close[i]
                    if close[i] < hwm - k * atr_i:
                        exit_now = True
                else:
                    if close[i] < lwm:
                        lwm = close[i]
                    if close[i] > lwm + k * atr_i:
                        exit_now = True
                if bars_held >= max_hold:
                    exit_now = True
                if exit_now:
                    pos = 0
                    bars_held = 0
                    sig[i] = 0
                else:
                    sig[i] = pos

        with np.errstate(divide="ignore", invalid="ignore"):
            raw = float(params.target_vol) / rvol
        raw = np.where(np.isfinite(raw), raw, 1.0)
        size = np.clip(raw, float(params.size_floor), float(params.size_cap))
        size = np.where(np.isfinite(size), size, 1.0)

        df = pd.DataFrame(index=data.index)
        df["signal"] = sig
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = size.astype(float)
        return SignalFrame(data=df, signal_column="signal", size_column="size")
