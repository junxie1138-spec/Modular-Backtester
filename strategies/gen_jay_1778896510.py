from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class Params:
    compress_thresh: float = 0.55
    prey_thresh: float = -0.015
    hold_bars: int = 15
    base_size: float = 0.95


class GeneratedStrategy(BaseStrategy[Params]):
    strategy_id = "gen_jay_1778896510"

    @classmethod
    def params_type(cls) -> type[Params]:
        return Params

    @staticmethod
    def warmup_bars(params: Params) -> int:
        # pct_change(1) before rolling(8) => 8 + 1 = 9
        return 9

    @staticmethod
    def indicators(data: pd.DataFrame, params: Params) -> pd.DataFrame:
        close = data["close"]
        cc_ret = close.pct_change(1)

        ret_range_5 = cc_ret.rolling(5).max() - cc_ret.rolling(5).min()
        ret_range_8 = cc_ret.rolling(8).max() - cc_ret.rolling(8).min()
        # ratio of short return-range to long return-range: low => coiling
        compression_ratio = ret_range_5 / ret_range_8.clip(lower=1e-8)
        cum_ret_5 = cc_ret.rolling(5).sum()

        ind = pd.DataFrame(index=data.index)
        ind["compression_ratio"] = compression_ratio
        ind["cum_ret_5"] = cum_ret_5
        return ind

    @staticmethod
    def generate_signals(
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: Params,
    ) -> SignalFrame:
        df = pd.DataFrame(index=data.index)

        compression_ratio = indicators["compression_ratio"]
        cum_ret_5 = indicators["cum_ret_5"]

        # Entry: return distribution coiling AND predator (seller) exhaustion
        raw_entries = (
            (compression_ratio < params.compress_thresh)
            & (cum_ret_5 < params.prey_thresh)
        ).fillna(False).values

        n = len(df)
        signal_arr = np.zeros(n, dtype=int)
        exit_bar = -1

        for i in range(n):
            if i < exit_bar:
                # inside fixed-bar hold window
                signal_arr[i] = 1
            elif raw_entries[i]:
                # new entry: set exit exactly hold_bars later
                exit_bar = i + params.hold_bars
                signal_arr[i] = 1
            # else: signal_arr[i] stays 0

        # Mandatory one-bar shift: decision on bar N fills on bar N+1
        df["signal"] = pd.Series(signal_arr, index=data.index).shift(1).fillna(0).astype(int)
        df["size"] = params.base_size

        return SignalFrame(data=df, signal_column="signal", size_column="size")
