from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class Params:
    dd_lookback: int = 252
    dd_pct_threshold: float = 20.0
    memory_bars: int = 10
    atr_period: int = 14
    vol_lookback: int = 63
    vol_pct_threshold: float = 35.0
    target_vol: float = 0.12
    breakeven_pct: float = 2.0
    trail_atr_mult: float = 2.0
    max_hold: int = 21


class GeneratedStrategy(BaseStrategy[Params]):
    strategy_id = "gen_jay_1778885033"

    @classmethod
    def params_type(cls):
        return Params

    @staticmethod
    def warmup_bars(params: Params) -> int:
        rv_window = max(params.atr_period * 3, 21)
        return max(params.dd_lookback, params.vol_lookback) + rv_window + 5

    @staticmethod
    def indicators(data: pd.DataFrame, params: Params) -> pd.DataFrame:
        ind = pd.DataFrame(index=data.index)
        close = data["close"]
        high = data["high"]
        low = data["low"]

        rolling_high = close.rolling(params.dd_lookback).max()
        drawdown = (close - rolling_high) / rolling_high.replace(0, np.nan)
        dd_rank = drawdown.rolling(params.dd_lookback).rank(pct=True) * 100

        prev_close = close.shift(1)
        tr = pd.concat(
            [high - low, (high - prev_close).abs(), (low - prev_close).abs()],
            axis=1,
        ).max(axis=1)
        atr = tr.ewm(span=params.atr_period, min_periods=params.atr_period).mean()
        atr_rank = atr.rolling(params.vol_lookback).rank(pct=True) * 100

        rv_window = max(params.atr_period * 3, 21)
        realized_vol = close.pct_change().rolling(rv_window).std() * np.sqrt(252)

        ind["dd_rank"] = dd_rank
        ind["atr"] = atr
        ind["atr_rank"] = atr_rank
        ind["realized_vol"] = realized_vol
        return ind

    @staticmethod
    def generate_signals(
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: Params,
    ) -> SignalFrame:
        n = len(data)
        signal = np.zeros(n, dtype=int)
        size = np.full(n, 0.95)

        close = data["close"].values
        dd_rank = indicators["dd_rank"].values
        atr_rank = indicators["atr_rank"].values
        atr = indicators["atr"].values
        realized_vol = indicators["realized_vol"].values

        in_trade = False
        entry_price = 0.0
        stop_price = 0.0
        breakeven_triggered = False
        trail_high = 0.0
        bars_held = 0

        for i in range(n):
            if np.isnan(dd_rank[i]) or np.isnan(atr_rank[i]) or np.isnan(atr[i]):
                continue

            c = close[i]

            if in_trade:
                bars_held += 1
                if c > trail_high:
                    trail_high = c

                if not breakeven_triggered and c >= entry_price * (1.0 + params.breakeven_pct / 100.0):
                    stop_price = entry_price
                    breakeven_triggered = True

                if breakeven_triggered:
                    candidate = trail_high - params.trail_atr_mult * atr[i]
                    if candidate > stop_price:
                        stop_price = candidate

                if c <= stop_price or bars_held >= params.max_hold:
                    signal[i] = 0
                    in_trade = False
                    entry_price = 0.0
                    stop_price = 0.0
                    breakeven_triggered = False
                    trail_high = 0.0
                    bars_held = 0
                else:
                    signal[i] = 1
                    rv = realized_vol[i]
                    if not np.isnan(rv) and rv > 1e-8:
                        size[i] = float(np.clip(params.target_vol / rv, 0.1, 1.0))
            else:
                start_idx = max(0, i - params.memory_bars + 1)
                recent_dd = dd_rank[start_idx : i + 1]
                # NaN <= threshold evaluates False in numpy, so NaN-safe
                had_deep_dd = bool(np.any(recent_dd <= params.dd_pct_threshold))
                vol_compressing = bool(atr_rank[i] <= params.vol_pct_threshold)
                dd_recovering = bool(dd_rank[i] > params.dd_pct_threshold)

                if had_deep_dd and vol_compressing and dd_recovering:
                    signal[i] = 1
                    in_trade = True
                    entry_price = c
                    trail_high = c
                    stop_price = c - params.trail_atr_mult * atr[i]
                    breakeven_triggered = False
                    bars_held = 0

                    rv = realized_vol[i]
                    if not np.isnan(rv) and rv > 1e-8:
                        size[i] = float(np.clip(params.target_vol / rv, 0.1, 1.0))

        sig_s = pd.Series(signal, index=data.index).shift(1).fillna(0).astype(int)
        size_s = pd.Series(size, index=data.index).shift(1).fillna(0.95)

        df = pd.DataFrame({"signal": sig_s, "size": size_s}, index=data.index)
        return SignalFrame(data=df, signal_column="signal", size_column="size")
