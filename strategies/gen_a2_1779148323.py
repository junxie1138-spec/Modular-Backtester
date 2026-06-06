from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class GeneratedParams:
    dd_lookback: int = 40
    depth_threshold: float = 0.07
    atr_period: int = 14
    comp_short: int = 5
    comp_long: int = 30
    comp_threshold: float = 0.85
    stop_k: float = 2.5
    max_hold: int = 5
    size: float = 1.0


class GeneratedStrategy(BaseStrategy[GeneratedParams]):
    strategy_id = "gen_a2_1779148323"

    @classmethod
    def params_type(cls):
        return GeneratedParams

    @staticmethod
    def warmup_bars(params: GeneratedParams) -> int:
        return int(max(params.dd_lookback, params.comp_long,
                       params.atr_period, params.comp_short)) + 1

    @staticmethod
    def indicators(data: pd.DataFrame, params: GeneratedParams) -> pd.DataFrame:
        high = data["high"]
        low = data["low"]
        close = data["close"]
        prev_close = close.shift(1)

        tr = pd.concat(
            [
                (high - low).abs(),
                (high - prev_close).abs(),
                (low - prev_close).abs(),
            ],
            axis=1,
        ).max(axis=1)

        atr = tr.rolling(params.atr_period).mean()
        atr_short = tr.rolling(params.comp_short).mean()
        atr_long = tr.rolling(params.comp_long).mean()
        comp = atr_short / atr_long.replace(0.0, np.nan)

        roll_max = close.rolling(params.dd_lookback).max()
        roll_min = close.rolling(params.dd_lookback).min()
        dd = close / roll_max - 1.0
        ru = close / roll_min - 1.0

        out = pd.DataFrame(index=data.index)
        out["atr"] = atr
        out["comp"] = comp
        out["dd"] = dd
        out["ru"] = ru
        return out

    @staticmethod
    def generate_signals(
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: GeneratedParams,
    ) -> SignalFrame:
        n = len(data)
        close = data["close"].to_numpy(dtype=float)
        atr = indicators["atr"].to_numpy(dtype=float)
        comp = indicators["comp"].to_numpy(dtype=float)
        dd = indicators["dd"].to_numpy(dtype=float)
        ru = indicators["ru"].to_numpy(dtype=float)

        # Two-primitive AND: range-compression must agree with drawdown depth.
        compressed = comp < params.comp_threshold  # NaN -> False

        out_signal = np.zeros(n, dtype=int)
        pos = 0
        entry_price = 0.0
        entry_atr = 0.0
        entry_bar = 0
        thr = float(params.depth_threshold)
        k = float(params.stop_k)
        max_hold = int(params.max_hold)

        for i in range(n):
            if pos == 0:
                a = atr[i]
                ddi = dd[i]
                rui = ru[i]
                ready = (
                    not np.isnan(a)
                    and a > 0.0
                    and not np.isnan(ddi)
                    and not np.isnan(rui)
                    and bool(compressed[i])
                )
                if ready:
                    go_long = ddi <= -thr
                    go_short = rui >= thr
                    if go_long and go_short:
                        # both displacement primitives fire: take the deeper one
                        if abs(ddi) >= rui:
                            go_short = False
                        else:
                            go_long = False
                    if go_long:
                        pos = 1
                        entry_price = close[i]
                        entry_atr = a
                        entry_bar = i
                    elif go_short:
                        pos = -1
                        entry_price = close[i]
                        entry_atr = a
                        entry_bar = i
            else:
                held = i - entry_bar
                exit_now = False
                if pos == 1:
                    if close[i] < entry_price - k * entry_atr:
                        exit_now = True
                    elif held >= max_hold:
                        exit_now = True
                else:
                    if close[i] > entry_price + k * entry_atr:
                        exit_now = True
                    elif held >= max_hold:
                        exit_now = True
                if exit_now:
                    pos = 0
            out_signal[i] = pos

        df = pd.DataFrame(index=data.index)
        df["signal"] = pd.Series(out_signal, index=data.index).shift(1).fillna(0).astype(int)
        size = float(params.size)
        if size <= 0.0:
            size = 1.0
        df["size"] = size
        return SignalFrame(data=df, signal_column="signal", size_column="size")
