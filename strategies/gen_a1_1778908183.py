from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class Params:
    high_window: int = 60
    rank_window: int = 120
    pct_threshold: float = 0.90
    recover_ma: int = 5
    atr_len: int = 14
    atr_mult: float = 2.5
    max_hold: int = 3
    refractory_bars: int = 5
    base_size: float = 1.0


class GeneratedStrategy(BaseStrategy[Params]):
    strategy_id = "gen_a1_1778908183"

    @classmethod
    def params_type(cls):
        return Params

    @staticmethod
    def warmup_bars(params: Params) -> int:
        return int(params.high_window + params.rank_window + params.atr_len + 5)

    @staticmethod
    def indicators(data: pd.DataFrame, params: Params) -> pd.DataFrame:
        close = data["close"]
        high = data["high"]
        low = data["low"]

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

        roll_high = close.rolling(
            params.high_window, min_periods=params.high_window
        ).max()
        is_below = (close < roll_high).fillna(False)
        b = is_below.astype(float)
        csum = b.cumsum()
        reset = csum.where(~is_below).ffill().fillna(0.0)
        duration = csum - reset

        duration_rank = duration.rolling(
            params.rank_window, min_periods=params.rank_window
        ).rank(pct=True)

        recover_ma = close.rolling(
            params.recover_ma, min_periods=params.recover_ma
        ).mean()

        out = pd.DataFrame(index=data.index)
        out["atr"] = atr
        out["roll_high"] = roll_high
        out["duration"] = duration
        out["duration_rank"] = duration_rank
        out["recover_ma"] = recover_ma
        return out

    @staticmethod
    def generate_signals(
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: Params,
    ) -> SignalFrame:
        close = data["close"]
        n = len(data)

        atr = indicators["atr"].to_numpy(dtype=float)
        drank = indicators["duration_rank"].fillna(0.0).to_numpy(dtype=float)
        rma = indicators["recover_ma"].to_numpy(dtype=float)
        c = close.to_numpy(dtype=float)
        c_prev = close.shift(1).to_numpy(dtype=float)

        entry = (
            (drank >= params.pct_threshold)
            & (c > rma)
            & (c > c_prev)
            & np.isfinite(rma)
            & np.isfinite(atr)
        )

        sig = np.zeros(n, dtype=int)
        size = np.full(n, float(params.base_size), dtype=float)

        in_pos = False
        hwm = 0.0
        bars_held = 0
        entry_size = float(params.base_size)
        refractory_until = -1

        for i in range(n):
            if not in_pos:
                if i >= refractory_until and bool(entry[i]):
                    in_pos = True
                    hwm = c[i]
                    bars_held = 0
                    entry_size = float(params.base_size) * (0.5 + drank[i])
                    if not np.isfinite(entry_size) or entry_size <= 0.0:
                        entry_size = float(params.base_size)
                    sig[i] = 1
                    size[i] = entry_size
                else:
                    sig[i] = 0
            else:
                bars_held += 1
                if c[i] > hwm:
                    hwm = c[i]
                stop = hwm - params.atr_mult * atr[i]
                hit_stop = bool(np.isfinite(stop)) and c[i] < stop
                if hit_stop or bars_held >= params.max_hold:
                    in_pos = False
                    sig[i] = 0
                    size[i] = entry_size
                    refractory_until = i + int(params.refractory_bars)
                else:
                    sig[i] = 1
                    size[i] = entry_size

        df = pd.DataFrame(index=data.index)
        df["signal"] = sig
        df["size"] = size
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = df["size"].shift(1).fillna(float(params.base_size))
        df["size"] = df["size"].clip(lower=1e-6)
        return SignalFrame(data=df, signal_column="signal", size_column="size")
