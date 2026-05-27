from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class Params:
    ac_window: int = 20
    autocorr_threshold: float = 0.15
    snr_threshold: float = 1.0
    base_size: float = 1.0


class GeneratedStrategy(BaseStrategy[Params]):
    strategy_id = "gen_a1_1779652526"

    @classmethod
    def params_type(cls):
        return Params

    @classmethod
    def warmup_bars(cls, params: Params) -> int:
        return int(params.ac_window) + 2

    def indicators(self, data: pd.DataFrame, params: Params) -> pd.DataFrame:
        close = data["close"].astype(float)
        ret = close.pct_change()
        win = int(params.ac_window)
        if win < 2:
            win = 2

        r0 = ret
        r1 = ret.shift(1)

        mean0 = r0.rolling(win).mean()
        mean1 = r1.rolling(win).mean()
        var0 = r0.rolling(win).var(ddof=0)
        var1 = r1.rolling(win).var(ddof=0)
        cross = (r0 * r1).rolling(win).mean()
        cov = cross - (mean0 * mean1)

        denom = np.sqrt(var0 * var1)
        autocorr = (cov / denom).replace([np.inf, -np.inf], np.nan)
        autocorr = autocorr.where(denom > 0)

        snr = autocorr.abs() * float(np.sqrt(win))

        out = pd.DataFrame(
            {
                "ret": ret,
                "autocorr": autocorr,
                "snr": snr,
            },
            index=data.index,
        )
        return out

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: Params,
    ) -> SignalFrame:
        ac = indicators["autocorr"]
        snr = indicators["snr"]
        thresh = float(params.autocorr_threshold)
        snr_thresh = float(params.snr_threshold)

        entry_long = ((ac > thresh) & (snr > snr_thresh)).fillna(False).to_numpy()
        entry_exit = ((ac < -thresh) & (snr > snr_thresh)).fillna(False).to_numpy()

        n = len(data)
        raw = np.zeros(n, dtype=np.int8)
        pos = 0
        for i in range(n):
            if pos == 0:
                if entry_long[i]:
                    pos = 1
            else:
                if entry_exit[i]:
                    pos = 0
            raw[i] = pos

        df = pd.DataFrame(index=data.index)
        df["signal"] = pd.Series(raw, index=data.index).shift(1).fillna(0).astype(int)
        size_val = float(params.base_size)
        if size_val <= 0:
            size_val = 1.0
        df["size"] = size_val
        return SignalFrame(data=df, signal_column="signal", size_column="size")
