from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class Params:
    atr_len: int = 14
    vol_len: int = 10
    vol_lb: int = 60
    corr_len: int = 40
    shock_thr: float = 1.0
    entry_vol_floor: float = 0.3
    corr_neg_thr: float = -0.2
    be_trigger: float = 0.01
    init_atr_mult: float = 2.5
    trail_atr_mult: float = 2.0
    max_hold: int = 3
    size_base: float = 0.5
    size_scale: float = 0.5
    size_cap: float = 2.0


class GeneratedStrategy(BaseStrategy[Params]):
    strategy_id = "gen_a1_1778884923"

    @classmethod
    def params_type(cls):
        return Params

    @staticmethod
    def warmup_bars(params: Params) -> int:
        return int(max(
            params.vol_lb + params.vol_len,
            params.corr_len + params.vol_len,
            params.atr_len,
        ) + 5)

    @staticmethod
    def indicators(data: pd.DataFrame, params: Params) -> pd.DataFrame:
        close = data["close"]
        high = data["high"]
        low = data["low"]

        ret = close.pct_change()
        log_ret = np.log(close).diff()
        vol = log_ret.rolling(params.vol_len).std()
        vol_change = vol.diff()

        vol_mean = vol.rolling(params.vol_lb).mean()
        vol_std = vol.rolling(params.vol_lb).std()
        vol_z = (vol - vol_mean) / vol_std.replace(0.0, np.nan)

        lev_corr = ret.rolling(params.corr_len).corr(vol_change)

        prev_close = close.shift(1)
        tr = pd.concat([
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ], axis=1).max(axis=1)
        atr = tr.rolling(params.atr_len).mean()

        rz = ret.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        vz = vol_z.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        lc = lev_corr.replace([np.inf, -np.inf], np.nan).fillna(0.0)

        diffusive = lc <= params.corr_neg_thr
        propagating = ~diffusive
        shock = vz > params.shock_thr
        expand = vz > params.entry_vol_floor

        mr_long = diffusive & shock & (rz < 0.0)
        mr_short = diffusive & shock & (rz > 0.0)
        mo_long = propagating & expand & (rz > 0.0)
        mo_short = propagating & expand & (rz < 0.0)

        raw_sig = np.zeros(len(data), dtype=float)
        raw_sig[(mr_long | mo_long).to_numpy()] = 1.0
        raw_sig[(mr_short | mo_short).to_numpy()] = -1.0

        size_mult = params.size_base + params.size_scale * np.clip(
            vz.to_numpy() - params.entry_vol_floor, 0.0, None
        )
        size_mult = np.clip(size_mult, params.size_base, params.size_cap)

        out = pd.DataFrame(index=data.index)
        out["vol_z"] = vz
        out["lev_corr"] = lc
        out["atr"] = atr
        out["ret"] = rz
        out["raw_sig"] = raw_sig
        out["raw_size"] = size_mult
        return out

    @staticmethod
    def generate_signals(data: pd.DataFrame, indicators: pd.DataFrame,
                         ctx: StrategyContext, params: Params) -> SignalFrame:
        close = data["close"].to_numpy(dtype=float)
        atr = indicators["atr"].to_numpy(dtype=float)
        raw_sig = indicators["raw_sig"].to_numpy(dtype=float)
        raw_size = indicators["raw_size"].to_numpy(dtype=float)
        n = len(close)

        pos = np.zeros(n, dtype=int)
        size_arr = np.ones(n, dtype=float)

        cur_pos = 0
        entry_price = 0.0
        stop = 0.0
        be_armed = False
        bars_held = 0
        cur_size = 1.0

        for i in range(n):
            if cur_pos == 0:
                s = int(raw_sig[i])
                a = atr[i]
                if s != 0 and np.isfinite(a) and a > 0.0:
                    cur_pos = s
                    entry_price = close[i]
                    rs = raw_size[i]
                    cur_size = rs if (np.isfinite(rs) and rs > 0.0) else 1.0
                    be_armed = False
                    bars_held = 0
                    if cur_pos == 1:
                        stop = entry_price - params.init_atr_mult * a
                    else:
                        stop = entry_price + params.init_atr_mult * a
                    pos[i] = cur_pos
                    size_arr[i] = cur_size
                else:
                    pos[i] = 0
                    size_arr[i] = 1.0
            else:
                bars_held += 1
                price = close[i]
                a = atr[i] if (np.isfinite(atr[i]) and atr[i] > 0.0) else 0.0
                exit_now = False
                if cur_pos == 1:
                    if not be_armed and price >= entry_price * (1.0 + params.be_trigger):
                        be_armed = True
                        stop = max(stop, entry_price)
                    if be_armed and a > 0.0:
                        stop = max(stop, price - params.trail_atr_mult * a)
                    if price <= stop:
                        exit_now = True
                else:
                    if not be_armed and price <= entry_price * (1.0 - params.be_trigger):
                        be_armed = True
                        stop = min(stop, entry_price)
                    if be_armed and a > 0.0:
                        stop = min(stop, price + params.trail_atr_mult * a)
                    if price >= stop:
                        exit_now = True
                if bars_held >= params.max_hold:
                    exit_now = True
                if exit_now:
                    cur_pos = 0
                    pos[i] = 0
                    size_arr[i] = 1.0
                else:
                    pos[i] = cur_pos
                    size_arr[i] = cur_size

        df = pd.DataFrame(index=data.index)
        df["signal"] = pos
        df["size"] = size_arr
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = df["size"].shift(1).fillna(1.0)
        df["size"] = df["size"].clip(lower=0.01)
        return SignalFrame(data=df, signal_column="signal", size_column="size")
