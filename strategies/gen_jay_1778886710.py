from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class Params:
    atr_period: int = 14
    eq_period: int = 50
    pct_window: int = 126
    entry_pct_lo: float = 0.05
    entry_pct_hi: float = 0.25
    stop_k: float = 2.0
    max_hold: int = 10
    trade_size: float = 1.0


class GeneratedStrategy(BaseStrategy[Params]):
    strategy_id = "gen_jay_1778886710"

    @classmethod
    def params_type(cls):
        return Params

    @staticmethod
    def warmup_bars(params: Params) -> int:
        return params.pct_window + params.eq_period + params.atr_period + 5

    @staticmethod
    def indicators(data: pd.DataFrame, params: Params) -> pd.DataFrame:
        ind = pd.DataFrame(index=data.index)
        close = data["close"]
        high = data["high"]
        low = data["low"]

        prev_close = close.shift(1)
        tr = pd.concat(
            [
                high - low,
                (high - prev_close).abs(),
                (low - prev_close).abs(),
            ],
            axis=1,
        ).max(axis=1)
        ind["atr"] = tr.rolling(params.atr_period).mean()

        ind["eq"] = close.rolling(params.eq_period).mean()

        atr_safe = ind["atr"].replace(0, np.nan)
        ind["displacement"] = (close - ind["eq"]) / atr_safe

        ind["disp_pct"] = (
            ind["displacement"]
            .rolling(params.pct_window)
            .rank(pct=True)
        )

        return ind

    @staticmethod
    def generate_signals(
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: Params,
    ) -> SignalFrame:
        close = data["close"].values
        disp_pct = indicators["disp_pct"].values
        atr = indicators["atr"].values

        n = len(close)
        signal = np.zeros(n, dtype=int)
        size = np.full(n, params.trade_size, dtype=float)

        in_trade = False
        entry_price = 0.0
        entry_atr = 0.0
        bars_held = 0

        for i in range(n):
            if np.isnan(disp_pct[i]) or np.isnan(atr[i]) or np.isnan(close[i]):
                continue

            if in_trade:
                bars_held += 1
                stop_price = entry_price - params.stop_k * entry_atr
                if close[i] <= stop_price or bars_held >= params.max_hold:
                    in_trade = False
                    bars_held = 0
                else:
                    signal[i] = 1
            else:
                if params.entry_pct_lo <= disp_pct[i] <= params.entry_pct_hi:
                    in_trade = True
                    entry_price = close[i]
                    entry_atr = atr[i]
                    bars_held = 0
                    signal[i] = 1

        df = pd.DataFrame({"signal": signal, "size": size}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)

        return SignalFrame(data=df, signal_column="signal", size_column="size")
