from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class NoisyDislocationParams:
    rank_window: int = 5
    snr_window: int = 5
    entry_rank_max: float = 0.20
    entry_snr_max: float = 0.30
    hold_bars: int = 2
    size: float = 1.0


class GeneratedStrategy(BaseStrategy[NoisyDislocationParams]):
    strategy_id = "gen_a1_1779647756"

    @classmethod
    def params_type(cls) -> type[NoisyDislocationParams]:
        return NoisyDislocationParams

    @classmethod
    def warmup_bars(cls, params: NoisyDislocationParams) -> int:
        rank_w = max(int(params.rank_window), 2)
        snr_w = max(int(params.snr_window), 2)
        return int(max(rank_w, snr_w + 1))

    def indicators(self, data: pd.DataFrame, params: NoisyDislocationParams) -> pd.DataFrame:
        close = data["close"].astype(float)
        ret = close.pct_change()

        rank_w = max(int(params.rank_window), 2)
        snr_w = max(int(params.snr_window), 2)

        close_rank = close.rolling(rank_w, min_periods=rank_w).rank(pct=True)

        roll_mean = ret.rolling(snr_w, min_periods=snr_w).mean()
        roll_std = ret.rolling(snr_w, min_periods=snr_w).std(ddof=0)
        denom = roll_std.replace(0.0, np.nan)
        snr = (roll_mean.abs() / denom).replace([np.inf, -np.inf], np.nan)

        out = pd.DataFrame(index=data.index)
        out["close_rank"] = close_rank
        out["snr"] = snr
        return out

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: NoisyDislocationParams,
    ) -> SignalFrame:
        close_rank = indicators["close_rank"].to_numpy()
        snr = indicators["snr"].to_numpy()

        rank_max = float(params.entry_rank_max)
        snr_max = float(params.entry_snr_max)
        hold = max(int(params.hold_bars), 1)

        n = len(data)
        raw = np.zeros(n, dtype=np.int64)
        hold_left = 0
        for i in range(n):
            if hold_left > 0:
                raw[i] = 1
                hold_left -= 1
                continue
            cr = close_rank[i]
            sn = snr[i]
            if not (np.isfinite(cr) and np.isfinite(sn)):
                continue
            if cr <= rank_max and sn <= snr_max:
                raw[i] = 1
                hold_left = hold - 1

        signal = pd.Series(raw, index=data.index).shift(1).fillna(0).astype(int)
        size_val = float(params.size)
        if size_val <= 0.0:
            size_val = 1.0
        size = pd.Series(size_val, index=data.index, dtype=float)

        df = pd.DataFrame({"signal": signal, "size": size}, index=data.index)
        return SignalFrame(data=df, signal_column="signal", size_column="size")
