from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class HarmonicNodeParams:
    base_period: int = 5
    compression_pct: float = 30.0
    rank_window: int = 60
    profit_target_pct: float = 2.0
    time_stop_bars: int = 5


class GeneratedStrategy(BaseStrategy[HarmonicNodeParams]):
    strategy_id = "gen_jay_1778904130"

    @classmethod
    def params_type(cls):
        return HarmonicNodeParams

    @staticmethod
    def warmup_bars(params: HarmonicNodeParams) -> int:
        return params.rank_window + params.base_period * 4 + 5

    @staticmethod
    def indicators(data: pd.DataFrame, params: HarmonicNodeParams) -> pd.DataFrame:
        ind = pd.DataFrame(index=data.index)

        high = data["high"]
        low = data["low"]
        close = data["close"]
        open_ = data["open"]

        prev_close = close.shift(1)
        tr = pd.concat(
            [
                high - low,
                (high - prev_close).abs(),
                (low - prev_close).abs(),
            ],
            axis=1,
        ).max(axis=1)

        p1 = params.base_period
        p2 = p1 * 2
        p4 = p1 * 4

        atr1 = tr.rolling(p1, min_periods=p1).mean()
        atr2 = tr.rolling(p2, min_periods=p2).mean()
        atr4 = tr.rolling(p4, min_periods=p4).mean()

        rw = params.rank_window

        def range_pct(s: pd.Series) -> pd.Series:
            lo = s.rolling(rw, min_periods=rw).min()
            hi = s.rolling(rw, min_periods=rw).max()
            span = (hi - lo).replace(0.0, np.nan)
            return (s - lo) / span * 100.0

        ind["r1"] = range_pct(atr1)
        ind["r2"] = range_pct(atr2)
        ind["r4"] = range_pct(atr4)

        cpct = params.compression_pct
        ind["node"] = (
            (ind["r1"] < cpct) & (ind["r2"] < cpct) & (ind["r4"] < cpct)
        ).astype(int)

        bullish = (close > open_).astype(int)
        ind["two_bar_confirm"] = (
            bullish.shift(1).fillna(0).astype(int)
            & bullish.shift(2).fillna(0).astype(int)
        )

        ind["raw_entry"] = (
            (ind["node"] == 1) & (ind["two_bar_confirm"] == 1)
        ).astype(int)

        return ind

    @staticmethod
    def generate_signals(
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: HarmonicNodeParams,
    ) -> SignalFrame:
        n = len(data)
        close = data["close"].values
        raw_entry = indicators["raw_entry"].values

        signal = np.zeros(n, dtype=int)
        size = np.full(n, 0.95)

        profit_mult = 1.0 + params.profit_target_pct / 100.0
        time_stop = int(params.time_stop_bars)

        in_trade = False
        entry_price = 0.0
        bars_held = 0

        for i in range(n):
            if in_trade:
                bars_held += 1
                if close[i] >= entry_price * profit_mult or bars_held >= time_stop:
                    in_trade = False
                    signal[i] = 0
                else:
                    signal[i] = 1
            else:
                if raw_entry[i] == 1:
                    in_trade = True
                    entry_price = close[i]
                    bars_held = 0
                    signal[i] = 1
                else:
                    signal[i] = 0

        df = pd.DataFrame({"signal": signal, "size": size}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)

        return SignalFrame(data=df, signal_column="signal", size_column="size")
