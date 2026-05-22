from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class Params:
    ma_len: int = 50
    z_len: int = 20
    pct_len: int = 252
    entry_pct: float = 0.70
    trend_len: int = 200
    atr_len: int = 14
    trail_k: float = 2.5
    vol_len: int = 20
    vol_target: float = 0.15
    spike_z: float = 2.5
    refractory_bars: int = 5
    max_size: float = 1.5
    min_size: float = 0.10


class GeneratedStrategy(BaseStrategy[Params]):
    strategy_id = "gen_a2_1779154269"

    @classmethod
    def params_type(cls):
        return Params

    @staticmethod
    def warmup_bars(params):
        base = params.ma_len + params.z_len + params.pct_len
        return int(max(base, params.trend_len, params.atr_len + 1, params.vol_len + 1) + 5)

    @staticmethod
    def indicators(data, params):
        close = data["close"]
        high = data["high"]
        low = data["low"]
        out = pd.DataFrame(index=data.index)

        ma = close.rolling(params.ma_len).mean()
        dist = (close - ma) / ma.replace(0.0, np.nan)
        z_mean = dist.rolling(params.z_len).mean()
        z_std = dist.rolling(params.z_len).std().replace(0.0, np.nan)
        z = (dist - z_mean) / z_std
        z_pct = z.rolling(params.pct_len).rank(pct=True)

        prev_close = close.shift(1)
        tr = pd.concat(
            [
                high - low,
                (high - prev_close).abs(),
                (low - prev_close).abs(),
            ],
            axis=1,
        ).max(axis=1)
        atr = tr.rolling(params.atr_len).mean()

        ret = close.pct_change()
        rv = ret.rolling(params.vol_len).std() * np.sqrt(252.0)
        ret_std = ret.rolling(params.vol_len).std().replace(0.0, np.nan)
        ret_z = ret / ret_std
        spike = (ret_z.abs() > params.spike_z).fillna(False)
        grp = spike.cumsum()
        since = spike.groupby(grp).cumcount()
        in_refractory = (grp > 0) & (since <= params.refractory_bars)

        trend = close > close.rolling(params.trend_len).mean()

        out["ma"] = ma
        out["z"] = z
        out["z_pct"] = z_pct
        out["atr"] = atr
        out["rv"] = rv
        out["in_refractory"] = in_refractory.astype(float)
        out["trend"] = trend.astype(float)
        return out

    @staticmethod
    def generate_signals(data, indicators, ctx, params):
        n = len(data)
        close = data["close"].to_numpy(dtype=float)
        atr = indicators["atr"].to_numpy(dtype=float)
        rv = indicators["rv"].to_numpy(dtype=float)

        z_pct = indicators["z_pct"]
        cross_up = (z_pct.shift(1) < params.entry_pct) & (z_pct >= params.entry_pct)
        trend_ok = indicators["trend"] > 0.5
        not_refr = indicators["in_refractory"] < 0.5
        arm = (cross_up & trend_ok & not_refr).fillna(False).to_numpy()

        raw = np.zeros(n, dtype=int)
        position = 0
        hwm = 0.0
        for i in range(n):
            a = atr[i]
            if position == 0:
                if arm[i] and not np.isnan(a) and a > 0.0:
                    position = 1
                    hwm = close[i]
                    raw[i] = 1
            else:
                c = close[i]
                if c > hwm:
                    hwm = c
                stop = hwm - params.trail_k * a if not np.isnan(a) else hwm
                if c <= stop:
                    position = 0
                    raw[i] = 0
                else:
                    raw[i] = 1

        with np.errstate(divide="ignore", invalid="ignore"):
            safe_rv = np.where((rv > 0.0) & ~np.isnan(rv), rv, np.nan)
            size = np.clip(params.vol_target / safe_rv, params.min_size, params.max_size)
        size = np.where(np.isnan(size), params.min_size, size)

        df = pd.DataFrame(index=data.index)
        df["signal"] = raw
        df["size"] = size.astype(float)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")
