from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class Params:
    atr_window: int = 14
    atr_slope_window: int = 5
    atr_lo_pct: float = 0.35
    atr_hi_pct: float = 0.75
    bull_window: int = 10
    bull_threshold: float = 0.60
    rank_window: int = 252
    be_pct: float = 0.02
    k_atr: float = 2.0
    spike_threshold: float = 0.85
    refractory_bars: int = 5


class GeneratedStrategy(BaseStrategy[Params]):
    strategy_id = "gen_jay_1778896654"

    @classmethod
    def params_type(cls) -> type[Params]:
        return Params

    @staticmethod
    def warmup_bars(params: Params) -> int:
        return params.rank_window + params.atr_window + params.atr_slope_window

    @staticmethod
    def indicators(data: pd.DataFrame, params: Params) -> pd.DataFrame:
        ind = pd.DataFrame(index=data.index)

        prev_close = data["close"].shift(1)
        tr = pd.concat(
            [
                data["high"] - data["low"],
                (data["high"] - prev_close).abs(),
                (data["low"] - prev_close).abs(),
            ],
            axis=1,
        ).max(axis=1)
        ind["atr"] = tr.rolling(params.atr_window).mean()
        ind["atr_rank"] = ind["atr"].rolling(params.rank_window).rank(pct=True)
        ind["atr_slope"] = ind["atr"].diff(params.atr_slope_window)

        bullish = (data["close"] > data["open"]).astype(float)
        ind["bull_ratio"] = bullish.rolling(params.bull_window).mean()

        return ind

    @staticmethod
    def generate_signals(
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: Params,
    ) -> SignalFrame:
        df = pd.DataFrame(index=data.index)

        atr_signal = (
            (indicators["atr_slope"] > 0)
            & (indicators["atr_rank"] >= params.atr_lo_pct)
            & (indicators["atr_rank"] <= params.atr_hi_pct)
        )
        bull_signal = indicators["bull_ratio"] >= params.bull_threshold
        raw_signal = (atr_signal & bull_signal).astype(int).values
        is_spike = (indicators["atr_rank"] > params.spike_threshold).astype(int).values

        closes = data["close"].values
        atrs = indicators["atr"].values
        n = len(df)
        signals = np.zeros(n, dtype=np.int64)

        in_trade = False
        entry_price = 0.0
        stop = 0.0
        be_triggered = False
        refractory_count = 0

        for i in range(n):
            close = closes[i]
            atr_val = atrs[i]

            if is_spike[i]:
                refractory_count = max(refractory_count, int(params.refractory_bars))

            if in_trade:
                if np.isnan(atr_val):
                    signals[i] = 0
                    in_trade = False
                    be_triggered = False
                else:
                    if not be_triggered and close >= entry_price * (1.0 + params.be_pct):
                        be_triggered = True
                        if entry_price > stop:
                            stop = entry_price

                    if be_triggered:
                        trail_stop = close - params.k_atr * atr_val
                        if trail_stop > stop:
                            stop = trail_stop

                    if close <= stop:
                        signals[i] = 0
                        in_trade = False
                        be_triggered = False
                    else:
                        signals[i] = 1
            else:
                if refractory_count > 0:
                    refractory_count -= 1

                if raw_signal[i] == 1 and refractory_count == 0 and not np.isnan(atr_val):
                    in_trade = True
                    entry_price = close
                    stop = close - params.k_atr * atr_val
                    be_triggered = False
                    signals[i] = 1

        df["signal"] = signals
        df["size"] = 1.0
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)

        return SignalFrame(data=df, signal_column="signal", size_column="size")
