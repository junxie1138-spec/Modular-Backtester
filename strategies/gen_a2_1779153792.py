from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class Params:
    ma_len: int = 50
    z_len: int = 50
    z_susc: float = -1.5
    z_entry: float = -0.4
    susc_lookback: int = 15
    roc_len: int = 5
    infect_smooth: int = 3
    trans_pct_len: int = 60
    trans_pct: float = 0.6
    atr_len: int = 14
    atr_k: float = 2.5
    vol_len: int = 20
    target_vol: float = 0.15
    max_leverage: float = 1.0
    min_size: float = 0.1


class GeneratedStrategy(BaseStrategy[Params]):
    strategy_id = "gen_a2_1779153792"

    @classmethod
    def params_type(cls) -> type[Params]:
        return Params

    @staticmethod
    def warmup_bars(params: Params) -> int:
        return int(max(
            params.ma_len,
            params.z_len,
            params.atr_len + 1,
            params.trans_pct_len + params.roc_len + params.infect_smooth,
            params.susc_lookback,
            params.vol_len + 1,
        )) + 5

    @staticmethod
    def indicators(data: pd.DataFrame, params: Params) -> pd.DataFrame:
        close = data["close"]
        high = data["high"]
        low = data["low"]
        out = pd.DataFrame(index=data.index)

        # distance-from-MA z-score (primary primitive)
        sma = close.rolling(params.ma_len, min_periods=params.ma_len).mean()
        std = close.rolling(params.z_len, min_periods=params.z_len).std(ddof=0)
        std_safe = std.replace(0.0, np.nan)
        z = (close - sma) / std_safe
        out["z"] = z

        # SI-epidemic transmission: susceptible pool * infection velocity
        susc = (-z).clip(lower=0.0)
        roc = close.pct_change(params.roc_len)
        infect = roc.rolling(
            params.infect_smooth, min_periods=params.infect_smooth
        ).mean().clip(lower=0.0)
        trans = susc * infect
        out["trans"] = trans
        trans_rank = trans.rolling(
            params.trans_pct_len, min_periods=params.trans_pct_len
        ).rank(pct=True)
        out["trans_rank"] = trans_rank

        # ATR for the rolling-high trailing stop
        prev_close = close.shift(1)
        tr = pd.concat([
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ], axis=1).max(axis=1)
        atr = tr.rolling(params.atr_len, min_periods=params.atr_len).mean()
        out["atr"] = atr

        # primitive A: fresh z cross-up through z_entry after recent deep displacement
        cross_up = (z > params.z_entry) & (z.shift(1) <= params.z_entry)
        deep = z.rolling(
            params.susc_lookback, min_periods=params.susc_lookback
        ).min().shift(1) < params.z_susc
        prim_a = (cross_up & deep).fillna(False)

        # primitive B: transmission rate in top historical band
        prim_b = (trans_rank > params.trans_pct).fillna(False)

        # two-primitive AND: both must agree
        out["entry_raw"] = (prim_a & prim_b).astype(float)

        # volatility-targeted position size
        rets = close.pct_change()
        rvol = rets.rolling(
            params.vol_len, min_periods=params.vol_len
        ).std(ddof=0) * np.sqrt(252.0)
        size = params.target_vol / rvol.replace(0.0, np.nan)
        size = size.clip(lower=params.min_size, upper=params.max_leverage)
        size = size.replace([np.inf, -np.inf], np.nan).fillna(params.min_size)
        out["size"] = size

        return out

    @staticmethod
    def generate_signals(
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: Params,
    ) -> SignalFrame:
        close = data["close"].to_numpy(dtype=float)
        atr = indicators["atr"].to_numpy(dtype=float)
        entry = indicators["entry_raw"].to_numpy(dtype=float) > 0.5
        n = len(close)
        raw = np.zeros(n, dtype=int)

        in_pos = False
        hw = 0.0  # in-trade high-water mark (highest close since entry)
        for i in range(n):
            if in_pos:
                c = close[i]
                if c > hw:
                    hw = c  # ratchet up only
                a = atr[i]
                if np.isfinite(a) and c < hw - params.atr_k * a:
                    in_pos = False
                    raw[i] = 0
                    continue
                raw[i] = 1
            else:
                if entry[i]:
                    in_pos = True
                    hw = close[i]
                    raw[i] = 1

        df = pd.DataFrame(index=data.index)
        df["signal"] = raw
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)

        size = indicators["size"].to_numpy(dtype=float)
        size = np.where(np.isfinite(size) & (size > 0.0), size, params.min_size)
        df["size"] = size

        return SignalFrame(data=df, signal_column="signal", size_column="size")
