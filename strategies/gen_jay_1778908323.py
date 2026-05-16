from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class Params:
    vol_surge_mult: float = 1.4
    eom_days: int = 6


class GeneratedStrategy(BaseStrategy["Params"]):
    strategy_id = "gen_jay_1778908323"

    @classmethod
    def params_type(cls):
        return Params

    @staticmethod
    def warmup_bars(params: Params) -> int:
        # 20-bar vol median + 4-bar lookback for prior shock detection
        return 26

    @staticmethod
    def indicators(data: pd.DataFrame, params: Params) -> pd.DataFrame:
        ind = pd.DataFrame(index=data.index)
        close = data["close"]
        open_ = data["open"]
        volume = data["volume"]

        # 20-bar rolling volume median; vol_ratio is NaN during warmup
        vol_med = volume.rolling(20).median()
        vol_ratio = volume / vol_med.replace(0, np.nan)
        ind["vol_ratio"] = vol_ratio

        # Sell shockwave: high-volume bearish close (the 'traffic jam')
        sell_shock = (vol_ratio >= 1.2) & (close < open_)

        # Prior shock: did a sell shockwave occur in the last 1-4 bars?
        # (the wave has passed through; sellers now exhausted)
        ind["prior_sell_shock"] = (
            sell_shock.shift(1).fillna(False)
            | sell_shock.shift(2).fillna(False)
            | sell_shock.shift(3).fillna(False)
            | sell_shock.shift(4).fillna(False)
        ).astype(int)

        # Current bar is bullish
        ind["bullish_bar"] = (close > open_).astype(int)

        # Seasonality gate: last eom_days trading days of month OR first 2 days
        ym = data.index.to_period("M")
        day_rank = data.groupby(ym).cumcount() + 1          # 1-based rank within month
        month_size = day_rank.groupby(ym).transform("max")  # total trading days in month

        ind["eom_gate"] = (
            ((month_size - day_rank) < params.eom_days) | (day_rank <= 2)
        ).astype(int)

        return ind

    @staticmethod
    def generate_signals(
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: Params,
    ) -> SignalFrame:
        df = pd.DataFrame(index=data.index)

        # Entry: seasonal window + prior sell-shockwave + current vol-surge bullish bar
        # (traffic jam formed, wave dissipated, flow resumes with a buy surge)
        entry_cond = (
            (indicators["eom_gate"] == 1)
            & (indicators["prior_sell_shock"] == 1)
            & (indicators["vol_ratio"] >= params.vol_surge_mult)
            & (indicators["bullish_bar"] == 1)
        ).fillna(False)

        # Signal-reversal exit: hold (1) while all conditions remain True; exit (0) when any flips
        raw_signal = entry_cond.astype(int)

        # Mandatory 1-bar shift: decision on bar N, fill executes on bar N+1
        df["signal"] = raw_signal.shift(1).fillna(0).astype(int)
        df["size"] = 1.0

        return SignalFrame(data=df, signal_column="signal", size_column="size")
