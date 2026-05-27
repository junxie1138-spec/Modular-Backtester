from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class Params:
    rank_window: int = 60
    ma_window: int = 20
    ret_window: int = 10
    atr_window: int = 14
    long_pct: float = 0.10
    short_pct: float = 0.90
    atr_mult: float = 3.0
    allow_short: bool = True


class GeneratedStrategy(BaseStrategy[Params]):
    strategy_id = "gen_a1_1779650050"

    @classmethod
    def params_type(cls):
        return Params

    def warmup_bars(self, params):
        return int(max(
            params.rank_window + params.ma_window,
            params.rank_window + params.ret_window,
            params.atr_window,
        ) + 2)

    def indicators(self, data, params):
        close = data["close"].astype(float)
        high = data["high"].astype(float)
        low = data["low"].astype(float)

        ma = close.rolling(params.ma_window, min_periods=params.ma_window).mean()
        std = close.rolling(params.ma_window, min_periods=params.ma_window).std()
        std_safe = std.where(std > 0.0, np.nan)
        zscore = (close - ma) / std_safe
        z_rank = zscore.rolling(
            params.rank_window, min_periods=params.rank_window
        ).rank(pct=True)

        cum_ret = close.pct_change(params.ret_window)
        ret_rank = cum_ret.rolling(
            params.rank_window, min_periods=params.rank_window
        ).rank(pct=True)

        prev_close = close.shift(1)
        tr = pd.concat(
            [
                (high - low),
                (high - prev_close).abs(),
                (low - prev_close).abs(),
            ],
            axis=1,
        ).max(axis=1)
        atr = tr.rolling(params.atr_window, min_periods=params.atr_window).mean()

        return pd.DataFrame(
            {
                "z_rank": z_rank,
                "ret_rank": ret_rank,
                "atr": atr,
            },
            index=data.index,
        )

    def generate_signals(self, data, indicators, ctx, params):
        n = len(data)
        close = data["close"].to_numpy(dtype=float)
        z_rank = indicators["z_rank"].to_numpy(dtype=float)
        ret_rank = indicators["ret_rank"].to_numpy(dtype=float)
        atr = indicators["atr"].to_numpy(dtype=float)

        long_pct = float(params.long_pct)
        short_pct = float(params.short_pct)
        atr_mult = float(params.atr_mult)
        allow_short = bool(params.allow_short)

        raw = np.zeros(n, dtype=np.int64)
        position = 0
        hwm = 0.0
        lwm = 0.0
        entry_atr = 0.0

        for i in range(n):
            c = close[i]
            a = atr[i]
            zr = z_rank[i]
            rr = ret_rank[i]

            if position == 1 and np.isfinite(c) and entry_atr > 0.0:
                if c > hwm:
                    hwm = c
                stop = hwm - atr_mult * entry_atr
                if c <= stop:
                    position = 0
                    hwm = 0.0
                    entry_atr = 0.0
            elif position == -1 and np.isfinite(c) and entry_atr > 0.0:
                if c < lwm:
                    lwm = c
                stop = lwm + atr_mult * entry_atr
                if c >= stop:
                    position = 0
                    lwm = 0.0
                    entry_atr = 0.0

            if (
                position == 0
                and np.isfinite(c)
                and np.isfinite(a)
                and a > 0.0
                and np.isfinite(zr)
                and np.isfinite(rr)
            ):
                long_fire = (zr <= long_pct) and (rr <= long_pct)
                short_fire = allow_short and (zr >= short_pct) and (rr >= short_pct)
                if long_fire:
                    position = 1
                    hwm = c
                    entry_atr = a
                elif short_fire:
                    position = -1
                    lwm = c
                    entry_atr = a

            raw[i] = position

        df = pd.DataFrame(
            {
                "signal": raw,
                "size": np.ones(n, dtype=float),
            },
            index=data.index,
        )
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = df["size"].astype(float)
        return SignalFrame(data=df, signal_column="signal", size_column="size")
