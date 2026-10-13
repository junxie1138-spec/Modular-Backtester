from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class GeneratedParams:
    rank_window: int = 63
    short_window: int = 3
    atr_window: int = 14
    tail_pct: float = 0.1
    stop_atr: float = 2.0
    max_holding_bars: int = 10


class GeneratedStrategy(BaseStrategy[GeneratedParams]):
    strategy_id = "gen_local_1790714035"

    @classmethod
    def params_type(cls):
        return GeneratedParams

    @classmethod
    def warmup_bars(cls, params: GeneratedParams) -> int:
        return max(params.rank_window + 1, params.short_window, params.atr_window)

    @classmethod
    def indicators(cls, data: pd.DataFrame, params: GeneratedParams) -> pd.DataFrame:
        ret5 = data["close"].pct_change(5)
        rank = ret5.rolling(params.rank_window).rank(pct=True)
        avg = data["close"].rolling(params.short_window).mean()
        prior_avg = avg.shift(1)
        cross_up = (data["close"] > avg) & (data["close"].shift(1) <= prior_avg)
        cross_down = (data["close"] < avg) & (data["close"].shift(1) >= prior_avg)
        tr = pd.concat(
            [
                data["high"] - data["low"],
                (data["high"] - data["close"].shift(1)).abs(),
                (data["low"] - data["close"].shift(1)).abs(),
            ],
            axis=1,
        ).max(axis=1)
        atr = tr.rolling(params.atr_window).mean()
        return pd.DataFrame(
            {
                "return_rank": rank,
                "cross_up": cross_up.fillna(False),
                "cross_down": cross_down.fillna(False),
                "atr": atr,
            },
            index=data.index,
        )

    @classmethod
    def generate_signals(
        cls,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: GeneratedParams,
    ) -> SignalFrame:
        rank = indicators["return_rank"]
        long_setup = (rank <= params.tail_pct) & indicators["cross_up"]
        short_setup = (rank >= 1.0 - params.tail_pct) & indicators["cross_down"]
        signal = pd.Series(0, index=data.index, dtype=int)
        position = 0
        entry_price = 0.0
        entry_atr = 0.0
        held = 0
        for i in range(len(data)):
            close = float(data["close"].iloc[i])
            atr = indicators["atr"].iloc[i]
            if position:
                held += 1
                stopped = close <= entry_price - params.stop_atr * entry_atr if position > 0 else close >= entry_price + params.stop_atr * entry_atr
                if stopped or held >= params.max_holding_bars or (position > 0 and bool(short_setup.iloc[i])) or (position < 0 and bool(long_setup.iloc[i])):
                    position = 0
            if position == 0 and np.isfinite(atr) and atr > 0:
                if bool(long_setup.iloc[i]):
                    position, entry_price, entry_atr, held = 1, close, float(atr), 0
                elif bool(short_setup.iloc[i]):
                    position, entry_price, entry_atr, held = -1, close, float(atr), 0
            signal.iloc[i] = position
        df = pd.DataFrame(index=data.index)
        df["signal"] = signal.shift(1).fillna(0).astype(int)
        df["size"] = 1.0
        return SignalFrame(data=df, signal_column="signal", size_column="size")
