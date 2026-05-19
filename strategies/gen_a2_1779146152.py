from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class Params:
    rank_window: int = 40
    hold_bars: int = 2


class GeneratedStrategy(BaseStrategy[Params]):
    strategy_id = "gen_a2_1779146152"

    # Fixed (non-tunable) constants - keeps the strategy at <=2 tunable params.
    _DIP_PCTILE = 0.30      # close must rank below this within rank_window
    _TOM_HEAD = 3           # first N trading days of a month are 'turn of month'
    _TOM_TAIL = 2           # last N trading days of a month are 'turn of month'

    @classmethod
    def params_type(cls):
        return Params

    @staticmethod
    def warmup_bars(params: Params) -> int:
        return int(params.rank_window) + 1

    @staticmethod
    def indicators(data: pd.DataFrame, params: Params) -> pd.DataFrame:
        w = max(int(params.rank_window), 2)
        close = data["close"]

        # Rolling percentile rank of the current close within its trailing
        # window. Low value => price is depressed relative to recent range.
        pct_rank = close.rolling(w).rank(pct=True)

        # Trading-day-of-month accounting from the datetime index alone.
        idx = data.index
        ym = idx.to_period("M")
        helper = pd.Series(np.arange(len(data)), index=idx)
        tdom = helper.groupby(ym).cumcount() + 1
        month_size = helper.groupby(ym).transform("size")
        rdom = month_size - tdom + 1  # 1 == last trading day of the month

        tom_window = (tdom <= GeneratedStrategy._TOM_HEAD) | (
            rdom <= GeneratedStrategy._TOM_TAIL
        )

        out = pd.DataFrame(index=idx)
        out["pct_rank"] = pct_rank
        out["tom_window"] = tom_window.astype(float)
        return out

    @staticmethod
    def generate_signals(data, indicators, ctx, params):
        n = len(data)
        hold = max(int(params.hold_bars), 1)

        pct_rank = indicators["pct_rank"].to_numpy(dtype=float)
        tom = indicators["tom_window"].to_numpy(dtype=float) > 0.5

        # NaN-safe: comparison against NaN yields False during warmup.
        dip = np.where(np.isnan(pct_rank), False, pct_rank < GeneratedStrategy._DIP_PCTILE)
        entry = tom & dip

        # Fixed-bar exit: once an entry fires, stay long exactly `hold` bars,
        # then flatten. No re-entry until that holding period completes.
        sig = np.zeros(n, dtype=int)
        i = 0
        while i < n:
            if entry[i]:
                end = min(i + hold, n)
                sig[i:end] = 1
                i = end
            else:
                i += 1

        df = pd.DataFrame(index=data.index)
        df["signal"] = sig
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = 1.0
        return SignalFrame(data=df, signal_column="signal", size_column="size")
