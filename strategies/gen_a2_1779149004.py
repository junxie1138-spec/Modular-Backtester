from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class GenA2Params:
    er_window: int = 20
    entry_threshold: float = 0.35
    ma_window: int = 200
    spike_window: int = 20
    spike_mult: float = 3.0
    refractory_bars: int = 3


class GeneratedStrategy(BaseStrategy[GenA2Params]):
    strategy_id = "gen_a2_1779149004"

    @classmethod
    def params_type(cls) -> type[GenA2Params]:
        return GenA2Params

    @staticmethod
    def warmup_bars(params: GenA2Params) -> int:
        return int(max(params.ma_window,
                       params.er_window + 1,
                       params.spike_window + 1))

    @staticmethod
    def indicators(data: pd.DataFrame, params: GenA2Params) -> pd.DataFrame:
        close = data["close"].astype(float)
        out = pd.DataFrame(index=data.index)

        er_w = max(int(params.er_window), 2)
        ret = close.diff()
        net = close.diff(er_w)
        path = ret.abs().rolling(er_w, min_periods=er_w).sum()
        er = net / path
        er = er.where(path > 0.0, 0.0)
        out["er"] = er.fillna(0.0)

        ma_w = max(int(params.ma_window), 2)
        ma = close.rolling(ma_w, min_periods=ma_w).mean()
        out["above_ma"] = (close > ma).fillna(False)

        sp_w = max(int(params.spike_window), 2)
        ret_std = ret.rolling(sp_w, min_periods=sp_w).std()
        spike = ret.abs() > (float(params.spike_mult) * ret_std)
        out["spike"] = spike.fillna(False)

        return out

    @staticmethod
    def generate_signals(data: pd.DataFrame,
                         indicators: pd.DataFrame,
                         ctx: StrategyContext,
                         params: GenA2Params) -> SignalFrame:
        df = pd.DataFrame(index=data.index)

        er = indicators["er"].to_numpy(dtype=float)
        above_ma = indicators["above_ma"].to_numpy(dtype=bool)
        spike = indicators["spike"].to_numpy(dtype=bool)

        n = len(df)
        sig = np.zeros(n, dtype=int)
        thr = float(params.entry_threshold)
        ref_len = max(int(params.refractory_bars), 0)

        pos = 0
        refractory = 0
        for i in range(n):
            long_entry = bool(er[i] > thr) and bool(above_ma[i])
            short_entry = bool(er[i] < -thr) and (not bool(above_ma[i]))
            blocked = refractory > 0

            if pos == 0:
                if not blocked:
                    if long_entry:
                        pos = 1
                    elif short_entry:
                        pos = -1
            elif pos == 1:
                if short_entry and not blocked:
                    pos = -1
            else:
                if long_entry and not blocked:
                    pos = 1

            sig[i] = pos

            if refractory > 0:
                refractory -= 1
            if bool(spike[i]):
                refractory = ref_len

        df["signal"] = sig
        df["size"] = 1.0
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)

        return SignalFrame(data=df, signal_column="signal", size_column="size")
