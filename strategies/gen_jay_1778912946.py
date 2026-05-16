from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class Params:
    short_rank_window: int = 10
    long_rank_window: int = 50
    rank_gap_threshold: float = 0.20
    profit_target_pct: float = 0.030
    time_stop_bars: int = 8


class GeneratedStrategy(BaseStrategy[Params]):
    strategy_id = "gen_jay_1778912946"

    @classmethod
    def params_type(cls) -> type[Params]:
        return Params

    @staticmethod
    def warmup_bars(params: Params) -> int:
        # long_rank_window non-NaN returns start at bar long_rank_window;
        # two-bar confirmation needs one more bar.
        return params.long_rank_window + 2

    @staticmethod
    def indicators(data: pd.DataFrame, params: Params) -> pd.DataFrame:
        ind = pd.DataFrame(index=data.index)
        ret = data["close"].pct_change()
        # Predator: fast-moving short-window rank (catches up quickly)
        ind["predator_rank"] = ret.rolling(params.short_rank_window).rank(pct=True)
        # Prey: slow-moving long-window rank (baseline momentum tempo)
        ind["prey_rank"] = ret.rolling(params.long_rank_window).rank(pct=True)
        # Positive gap => predator overtook prey => bullish momentum acceleration
        ind["rank_gap"] = ind["predator_rank"] - ind["prey_rank"]
        return ind

    @staticmethod
    def generate_signals(
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: Params,
    ) -> SignalFrame:
        close = data["close"]
        gap = indicators["rank_gap"]
        n = len(data)

        # NaN comparisons return False, so warmup bars are naturally excluded
        raw_long = gap > params.rank_gap_threshold
        raw_short = gap < -params.rank_gap_threshold

        # Two-bar confirmation: condition must hold on bar i AND bar i-1
        confirmed_long = raw_long & raw_long.shift(1).fillna(False)
        confirmed_short = raw_short & raw_short.shift(1).fillna(False)

        signal_arr = np.zeros(n, dtype=int)
        size_arr = np.full(n, 0.95, dtype=float)

        in_trade = False
        trade_dir = 0
        entry_price = 0.0
        entry_bar = 0

        for i in range(n):
            if in_trade:
                bars_held = i - entry_bar
                pnl = (close.iloc[i] - entry_price) / entry_price * trade_dir
                if pnl >= params.profit_target_pct or bars_held >= params.time_stop_bars:
                    signal_arr[i] = 0
                    in_trade = False
                else:
                    signal_arr[i] = trade_dir
            else:
                if bool(confirmed_long.iloc[i]):
                    signal_arr[i] = 1
                    in_trade = True
                    trade_dir = 1
                    entry_price = float(close.iloc[i])
                    entry_bar = i
                elif bool(confirmed_short.iloc[i]):
                    signal_arr[i] = -1
                    in_trade = True
                    trade_dir = -1
                    entry_price = float(close.iloc[i])
                    entry_bar = i

        df = data.copy()
        df["signal"] = signal_arr
        df["size"] = size_arr
        # Mandatory 1-bar shift: decision on bar N fills on bar N+1
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)

        return SignalFrame(data=df, signal_column="signal", size_column="size")
