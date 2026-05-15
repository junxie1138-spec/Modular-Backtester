from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class Params:
    lookback: int = 8
    dd_thresh: float = 0.012
    short_dd_thresh: float = 0.004
    momentum_bars: int = 3
    trend_str_floor: float = -0.6
    profit_target_pct: float = 0.010
    time_stop_bars: int = 2


class GeneratedStrategy(BaseStrategy[Params]):
    strategy_id = "gen_jay_1778882259"

    @classmethod
    def params_type(cls):
        return Params

    @staticmethod
    def warmup_bars(params: Params) -> int:
        return max(params.lookback, params.momentum_bars) + 1

    @staticmethod
    def indicators(data: pd.DataFrame, params: Params) -> pd.DataFrame:
        ind = pd.DataFrame(index=data.index)

        roll_high = data["high"].rolling(params.lookback).max()
        roll_low = data["low"].rolling(params.lookback).min()

        safe_roll_high = roll_high.replace(0, np.nan)
        # Drawdown depth from rolling high: 0 = at high, positive = below high
        ind["dd_depth"] = (roll_high - data["close"]) / safe_roll_high

        # Net directional displacement vs full bar range (trend-strength proxy)
        total_range = (roll_high - roll_low).replace(0, np.nan)
        net_move = data["close"] - data["close"].shift(params.momentum_bars)
        ind["trend_str"] = net_move / total_range

        # Single-bar reversal confirmation
        ind["bar_chg"] = data["close"].pct_change(1)

        return ind

    @staticmethod
    def generate_signals(
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: Params,
    ) -> SignalFrame:
        df = pd.DataFrame(index=data.index)
        df["size"] = 1.0

        close = data["close"].to_numpy()
        dd = indicators["dd_depth"].to_numpy()
        trend_str = indicators["trend_str"].to_numpy()
        bar_chg = indicators["bar_chg"].to_numpy()

        n = len(df)

        # Vectorised entry conditions
        valid = ~(np.isnan(dd) | np.isnan(trend_str) | np.isnan(bar_chg))
        long_cond = (
            valid
            & (dd > params.dd_thresh)
            & (bar_chg > 0)
            & (trend_str > params.trend_str_floor)
        )
        short_cond = (
            valid
            & (dd < params.short_dd_thresh)
            & (bar_chg < 0)
            & (trend_str < 0)
        )
        raw = np.where(long_cond, 1, np.where(short_cond, -1, 0)).astype(int)

        # Path-dependent exit loop: profit-target + time-stop
        pre_signal = np.zeros(n, dtype=int)
        position = 0
        entry_price = 0.0
        bars_held = 0

        for i in range(n):
            if position != 0:
                bars_held += 1
                pnl = (close[i] - entry_price) / entry_price * position
                if pnl >= params.profit_target_pct or bars_held >= params.time_stop_bars:
                    pre_signal[i] = 0
                    position = 0
                    bars_held = 0
                    entry_price = 0.0
                else:
                    pre_signal[i] = position

            if position == 0 and raw[i] != 0:
                position = raw[i]
                entry_price = close[i]
                bars_held = 0
                pre_signal[i] = position

        df["signal"] = pd.Series(pre_signal, index=data.index)
        # Mandatory one-bar shift: decision on bar N executes on bar N+1
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)

        return SignalFrame(data=df, signal_column="signal", size_column="size")
