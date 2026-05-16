from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy

_GAP_RANK_WINDOW = 60
_ATR_WINDOW = 14


@dataclass(slots=True)
class GapRankReversalParams:
    gap_pct_thresh: float = 0.70
    atr_stop_mult: float = 2.5


class GeneratedStrategy(BaseStrategy["GapRankReversalParams"]):
    strategy_id = "gen_jay_1778910343"

    @classmethod
    def params_type(cls):
        return GapRankReversalParams

    @staticmethod
    def warmup_bars(params: GapRankReversalParams) -> int:
        return _GAP_RANK_WINDOW + _ATR_WINDOW + 2

    @staticmethod
    def indicators(data: pd.DataFrame, params: GapRankReversalParams) -> pd.DataFrame:
        ind = pd.DataFrame(index=data.index)

        prev_close = data["close"].shift(1)
        ind["gap"] = (data["open"] - prev_close) / prev_close

        abs_gap = ind["gap"].abs()
        ind["gap_rank"] = abs_gap.rolling(_GAP_RANK_WINDOW).rank(pct=True)

        tr = pd.concat(
            [
                data["high"] - data["low"],
                (data["high"] - prev_close).abs(),
                (data["low"] - prev_close).abs(),
            ],
            axis=1,
        ).max(axis=1)
        ind["atr"] = tr.rolling(_ATR_WINDOW).mean()

        return ind

    @staticmethod
    def generate_signals(
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: GapRankReversalParams,
    ) -> SignalFrame:
        n = len(data)
        sig = np.zeros(n, dtype=int)
        sz = np.ones(n, dtype=float)

        close = data["close"].to_numpy()
        open_ = data["open"].to_numpy()
        gap = indicators["gap"].to_numpy()
        gap_rank = indicators["gap_rank"].to_numpy()
        atr = indicators["atr"].to_numpy()

        in_trade = False
        hwm = 0.0

        for i in range(1, n):
            if np.isnan(gap_rank[i]) or np.isnan(atr[i]) or np.isnan(gap[i]):
                continue

            if in_trade:
                hwm = max(hwm, close[i])
                stop = hwm - params.atr_stop_mult * atr[i]
                if close[i] < stop:
                    sig[i] = 0
                    in_trade = False
                else:
                    sig[i] = 1
            else:
                is_down_gap = gap[i] < 0
                is_large_gap = gap_rank[i] >= params.gap_pct_thresh
                is_intraday_reversal = close[i] > open_[i]

                if is_down_gap and is_large_gap and is_intraday_reversal:
                    sig[i] = 1
                    in_trade = True
                    hwm = close[i]

        df = pd.DataFrame({"signal": sig, "size": sz}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)

        return SignalFrame(data=df, signal_column="signal", size_column="size")
