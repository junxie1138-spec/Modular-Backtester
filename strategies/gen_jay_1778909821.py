from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class Params:
    rp_window: int = 20
    vol_window: int = 20
    rp_low_thresh: float = 0.35
    vol_mult: float = 1.2
    profit_target: float = 0.03
    time_stop: int = 5


class GeneratedStrategy(BaseStrategy["Params"]):
    strategy_id = "gen_jay_1778909821"

    @classmethod
    def params_type(cls) -> type[Params]:
        return Params

    @staticmethod
    def warmup_bars(params: Params) -> int:
        return max(params.rp_window, params.vol_window) + 5

    @staticmethod
    def indicators(data: pd.DataFrame, params: Params) -> pd.DataFrame:
        ind = pd.DataFrame(index=data.index)
        hi = data["high"].rolling(params.rp_window).max()
        lo = data["low"].rolling(params.rp_window).min()
        rng = (hi - lo).replace(0, np.nan)
        ind["rp"] = (data["close"] - lo) / rng
        vol_ma = data["volume"].rolling(params.vol_window).mean()
        ind["vol_ratio"] = data["volume"] / vol_ma.replace(0, np.nan)
        return ind

    @staticmethod
    def generate_signals(
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: Params,
    ) -> SignalFrame:
        rp = indicators["rp"]
        vol_ratio = indicators["vol_ratio"]

        # Prey zone: price pinned at bottom of its rolling range
        in_prey = rp < params.rp_low_thresh

        # Volume-confirmed upward relative-position tick
        rp_up = rp.diff() > 0
        vol_ok = vol_ratio > params.vol_mult

        # Two-bar confirmation: both current and prior bar show vol-confirmed rp improvement
        bar_confirm = rp_up & vol_ok
        two_bar = bar_confirm & bar_confirm.shift(1).fillna(False)

        # Setup: price was in prey zone before the two confirmation bars began
        setup = in_prey.shift(2).fillna(False)

        raw_entry = (setup & two_bar).values.astype(int)
        closes = data["close"].values
        n = len(closes)

        signal_pre = np.zeros(n, dtype=int)
        in_trade = False
        entry_price = 0.0
        hold_bars = 0

        for i in range(n):
            if in_trade:
                hold_bars += 1
                pnl = (closes[i] - entry_price) / entry_price
                if pnl >= params.profit_target or hold_bars >= params.time_stop:
                    in_trade = False
                else:
                    signal_pre[i] = 1

            if (not in_trade) and raw_entry[i] == 1:
                in_trade = True
                entry_price = closes[i]
                hold_bars = 0
                signal_pre[i] = 1

        df = pd.DataFrame(index=data.index)
        df["signal"] = signal_pre
        df["size"] = 0.95
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)

        return SignalFrame(data=df, signal_column="signal", size_column="size")
