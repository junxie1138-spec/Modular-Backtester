from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class StandingWaveParams:
    rank_lookback: int = 60
    range_lookback: int = 60
    compression_window: int = 10
    entry_close_pct: float = 0.90
    exit_close_pct: float = 0.40
    compression_pct_max: float = 0.30
    size: float = 1.0


class GeneratedStrategy(BaseStrategy[StandingWaveParams]):
    strategy_id = "gen_a1_1779885405"

    @classmethod
    def params_type(cls):
        return StandingWaveParams

    def warmup_bars(self, params: StandingWaveParams) -> int:
        return int(max(params.rank_lookback, params.range_lookback) + params.compression_window + 2)

    def indicators(self, data: pd.DataFrame, params: StandingWaveParams) -> pd.DataFrame:
        close = data["close"].astype(float)
        high = data["high"].astype(float)
        low = data["low"].astype(float)

        rng = (high - low).clip(lower=0.0)

        rl = int(max(2, params.rank_lookback))
        gl = int(max(2, params.range_lookback))
        cw = int(max(1, params.compression_window))

        close_rank = close.rolling(rl, min_periods=rl).rank(pct=True)
        range_rank = rng.rolling(gl, min_periods=gl).rank(pct=True)
        compression = range_rank.rolling(cw, min_periods=cw).median()

        ind = pd.DataFrame(
            {
                "close_rank": close_rank,
                "range_rank": range_rank,
                "compression": compression,
            },
            index=data.index,
        )
        return ind

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: StandingWaveParams,
    ) -> SignalFrame:
        close_rank = indicators["close_rank"]
        compression = indicators["compression"]

        entry_cond = (close_rank > float(params.entry_close_pct)) & (
            compression < float(params.compression_pct_max)
        )
        exit_cond = close_rank < float(params.exit_close_pct)

        entry_arr = entry_cond.fillna(False).to_numpy(dtype=bool)
        exit_arr = exit_cond.fillna(False).to_numpy(dtype=bool)

        n = len(data)
        raw = np.zeros(n, dtype=np.int64)
        in_pos = False
        for i in range(n):
            if in_pos:
                if exit_arr[i]:
                    in_pos = False
                    raw[i] = 0
                else:
                    raw[i] = 1
            else:
                if entry_arr[i]:
                    in_pos = True
                    raw[i] = 1
                else:
                    raw[i] = 0

        sig = pd.Series(raw, index=data.index)
        size_series = pd.Series(float(max(params.size, 1e-9)), index=data.index)

        df = pd.DataFrame(
            {
                "signal": sig.shift(1).fillna(0).astype(int),
                "size": size_series,
            },
            index=data.index,
        )

        return SignalFrame(data=df, signal_column="signal", size_column="size")
