from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class Params:
    gap_window: int = 10
    rank_window: int = 252
    entry_pct: float = 0.72
    exit_pct: float = 0.40


class GeneratedStrategy(BaseStrategy["Params"]):
    strategy_id = "gen_jay_1778890888"

    @classmethod
    def params_type(cls):
        return Params

    @staticmethod
    def warmup_bars(params: Params) -> int:
        # prev_close shift (+1) + gap_window for rolling sum + rank_window for rolling rank
        return params.rank_window + params.gap_window + 1

    @staticmethod
    def indicators(data: pd.DataFrame, params: Params) -> pd.DataFrame:
        ind = pd.DataFrame(index=data.index)

        prev_close = data["close"].shift(1)
        gap = (data["open"] - prev_close) / prev_close

        # Cumulative gap pressure: rolling sum of normalized gaps over gap_window bars
        cum_gap = gap.rolling(params.gap_window, min_periods=params.gap_window).sum()

        # Percentile rank of cum_gap within its own rank_window history (the hard twist)
        cum_gap_rank = cum_gap.rolling(
            params.rank_window, min_periods=params.rank_window
        ).rank(pct=True)

        ind["cum_gap"] = cum_gap
        ind["cum_gap_rank"] = cum_gap_rank

        return ind

    @staticmethod
    def generate_signals(
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: Params,
    ) -> SignalFrame:
        df = pd.DataFrame(index=data.index)

        rank = indicators["cum_gap_rank"]
        cum_gap = indicators["cum_gap"]

        n = len(df)
        raw_signal = np.zeros(n, dtype=int)
        raw_size = np.ones(n, dtype=float)
        in_trade = False

        for i in range(n):
            r = rank.iloc[i]
            cg = cum_gap.iloc[i]

            if pd.isna(r) or pd.isna(cg):
                raw_signal[i] = 0
                raw_size[i] = 1.0
                continue

            if not in_trade:
                # Entry: plastic deformation threshold breached with positive direction
                if r >= params.entry_pct and cg > 0.0:
                    in_trade = True
                    raw_signal[i] = 1
                    raw_size[i] = float(r)  # size proportional to deformation strength
                else:
                    raw_signal[i] = 0
                    raw_size[i] = 1.0
            else:
                # Signal-reversal exit: gap pressure subsides or turns negative
                if r < params.exit_pct or cg <= 0.0:
                    in_trade = False
                    raw_signal[i] = 0
                    raw_size[i] = 1.0
                else:
                    raw_signal[i] = 1
                    raw_size[i] = float(r)

        df["signal"] = raw_signal
        df["size"] = raw_size

        # Mandatory 1-bar shift: decided on bar N close, filled on bar N+1
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = df["size"].shift(1).fillna(1.0)
        df.loc[df["size"] <= 0.0, "size"] = 1.0

        return SignalFrame(data=df, signal_column="signal", size_column="size")
