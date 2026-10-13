from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class QueueParams:
    backlog_window: int = 5
    capacity_window: int = 30
    overflow_mult: float = 1.6
    atr_window: int = 14
    breakeven_pct: float = 0.012
    atr_mult: float = 1.8
    max_hold: int = 5


class GeneratedStrategy(BaseStrategy[QueueParams]):
    strategy_id = "gen_local_1791301094"

    @classmethod
    def params_type(cls) -> type[QueueParams]:
        return QueueParams

    @staticmethod
    def warmup_bars(params: QueueParams) -> int:
        return max(params.backlog_window + 2, params.capacity_window + 2, params.atr_window + 1)

    @staticmethod
    def indicators(data: pd.DataFrame, params: QueueParams) -> pd.DataFrame:
        returns = data["close"].pct_change().replace([np.inf, -np.inf], np.nan).fillna(0.0)
        backlog = (-returns.clip(upper=0)).rolling(params.backlog_window).sum()
        capacity = returns.abs().rolling(params.capacity_window).mean() * params.backlog_window * params.overflow_mult
        previous_close = data["close"].shift(1).fillna(data["close"])
        true_range = pd.concat(
            [data["high"] - data["low"], (data["high"] - previous_close).abs(), (data["low"] - previous_close).abs()],
            axis=1,
        ).max(axis=1)
        atr = true_range.rolling(params.atr_window).mean()
        result = pd.DataFrame(index=data.index)
        result["returns"] = returns
        result["backlog"] = backlog.fillna(0.0)
        result["capacity"] = capacity.fillna(0.0)
        result["overflow"] = (backlog > capacity).fillna(False) & (capacity > 0).fillna(False)
        result["atr"] = atr.fillna(0.0)
        return result

    @staticmethod
    def generate_signals(data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: QueueParams) -> SignalFrame:
        raw = np.zeros(len(data), dtype=int)
        opening = data["open"].to_numpy(dtype=float)
        lows = data["low"].to_numpy(dtype=float)
        closes = data["close"].to_numpy(dtype=float)
        returns = indicators["returns"].to_numpy(dtype=float)
        backlog = indicators["backlog"].to_numpy(dtype=float)
        capacity = indicators["capacity"].to_numpy(dtype=float)
        overflow = indicators["overflow"].to_numpy(dtype=bool)
        atr = indicators["atr"].to_numpy(dtype=float)
        entry_price = 0.0
        stop = 0.0
        held_bars = 0
        warmup = GeneratedStrategy.warmup_bars(params)

        for i in range(len(data)):
            holding = i > 0 and raw[i - 1] == 1
            if holding:
                if held_bars == 0:
                    entry_price = opening[i]
                    stop = 0.0
                held_bars += 1
                exit_now = (stop > 0.0 and lows[i] <= stop) or held_bars >= params.max_hold
                if not exit_now:
                    if closes[i] >= entry_price * (1.0 + params.breakeven_pct):
                        stop = max(stop, entry_price)
                    if stop > 0.0:
                        stop = max(stop, entry_price, closes[i] - params.atr_mult * atr[i])
                    raw[i] = 1
                else:
                    held_bars = 0
            elif i >= warmup:
                held_bars = 0
                recovering_overflow = overflow[i - 2] and returns[i - 1] > 0.0 and returns[i] > 0.0
                calm_pullback = (
                    not overflow[i - 2]
                    and backlog[i - 1] < capacity[i - 1]
                    and backlog[i] < capacity[i]
                    and returns[i - 1] < 0.0
                    and returns[i] < 0.0
                )
                if recovering_overflow or calm_pullback:
                    raw[i] = 1

        df = pd.DataFrame(index=data.index)
        df["signal"] = raw
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = 1.0
        return SignalFrame(data=df, signal_column="signal", size_column="size")
