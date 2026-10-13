from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class GeneratedParams:
    drawdown_lookback: int = 20
    roc_period: int = 3
    rank_window: int = 60
    entry_percentile: float = 0.90
    drawdown_min: float = 0.04
    atr_period: int = 14
    breakeven_pct: float = 0.02
    trail_atr: float = 2.0
    max_hold: int = 5


class GeneratedStrategy(BaseStrategy[GeneratedParams]):
    strategy_id = "gen_local_1790713648"

    @classmethod
    def params_type(cls):
        return GeneratedParams

    @classmethod
    def warmup_bars(cls, params: GeneratedParams) -> int:
        return max(params.drawdown_lookback, params.roc_period + params.rank_window, params.atr_period) + 1

    @classmethod
    def indicators(cls, data: pd.DataFrame, params: GeneratedParams) -> pd.DataFrame:
        close = data["close"].astype(float)
        roc = close.pct_change(params.roc_period)
        acceleration = roc.diff()
        accel_rank = acceleration.rolling(params.rank_window, min_periods=params.rank_window).rank(pct=True)
        peak = close.rolling(params.drawdown_lookback, min_periods=params.drawdown_lookback).max()
        drawdown = close.div(peak).sub(1.0)
        tr = pd.concat([
            data["high"] - data["low"],
            (data["high"] - data["close"].shift(1)).abs(),
            (data["low"] - data["close"].shift(1)).abs(),
        ], axis=1).max(axis=1)
        atr = tr.rolling(params.atr_period, min_periods=params.atr_period).mean()
        return pd.DataFrame({"accel_rank": accel_rank, "drawdown": drawdown, "atr": atr}, index=data.index)

    @classmethod
    def generate_signals(cls, data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: GeneratedParams) -> SignalFrame:
        ready = indicators[["accel_rank", "drawdown", "atr"]].notna().all(axis=1)
        entry = ready & (indicators["drawdown"] <= -params.drawdown_min) & (indicators["accel_rank"] >= params.entry_percentile)
        raw_signal = pd.Series(0, index=data.index, dtype=int)
        raw_size = pd.Series(0.95, index=data.index, dtype=float)
        active = False
        entry_price = 0.0
        stop = 0.0
        highest_close = 0.0
        held = 0
        for i in range(len(data)):
            price = float(data["close"].iloc[i])
            atr = float(indicators["atr"].iloc[i]) if ready.iloc[i] else np.nan
            if active:
                held += 1
                if np.isfinite(atr):
                    highest_close = max(highest_close, price)
                    if highest_close >= entry_price * (1.0 + params.breakeven_pct):
                        stop = max(stop, entry_price, highest_close - params.trail_atr * atr)
                if price <= stop or held >= params.max_hold:
                    active = False
                else:
                    raw_signal.iloc[i] = 1
            elif bool(entry.iloc[i]) and np.isfinite(atr):
                active = True
                entry_price = price
                highest_close = price
                stop = entry_price - params.trail_atr * atr
                held = 0
                raw_signal.iloc[i] = 1
        df = pd.DataFrame(index=data.index)
        df["signal"] = raw_signal.shift(1).fillna(0).astype(int)
        df["size"] = raw_size
        return SignalFrame(data=df, signal_column="signal", size_column="size")
