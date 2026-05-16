from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class Params:
    ma_period: int = 20
    zscore_lookback: int = 60
    zscore_threshold: float = -1.5
    gap_lookback: int = 5
    gap_threshold: float = -0.003
    queue_capacity: int = 2
    atr_period: int = 14
    atr_stop_k: float = 2.0
    max_hold_bars: int = 2
    ma200_period: int = 200


class GeneratedStrategy(BaseStrategy[Params]):
    strategy_id = "gen_jay_1778910546"

    @classmethod
    def params_type(cls):
        return Params

    @staticmethod
    def warmup_bars(params: Params) -> int:
        return params.ma200_period + params.zscore_lookback + 5

    @staticmethod
    def indicators(data: pd.DataFrame, params: Params) -> pd.DataFrame:
        ind = pd.DataFrame(index=data.index)

        close = data["close"]
        high = data["high"]
        low = data["low"]
        open_ = data["open"]

        # MA-distance z-score: primary signal primitive
        ma = close.rolling(params.ma_period).mean()
        dist = close - ma
        dist_roll_mean = dist.rolling(params.zscore_lookback).mean()
        dist_roll_std = dist.rolling(params.zscore_lookback).std()
        ind["zscore"] = (dist - dist_roll_mean) / dist_roll_std.replace(0, np.nan)

        # Overnight gap as fraction of prior close
        prev_close = close.shift(1)
        gap_pct = (open_ - prev_close) / prev_close
        ind["gap_pct"] = gap_pct

        # Rolling count of meaningful downward gaps (queue fill level)
        down_gap_flag = (gap_pct < params.gap_threshold).astype(float)
        ind["gap_queue"] = down_gap_flag.rolling(params.gap_lookback).sum()

        # ATR for fixed volatility stop
        tr = pd.concat([
            high - low,
            (high - close.shift(1)).abs(),
            (low - close.shift(1)).abs(),
        ], axis=1).max(axis=1)
        ind["atr"] = tr.rolling(params.atr_period).mean()

        # 200-day MA for bull/bear regime filter
        ind["ma200"] = close.rolling(params.ma200_period).mean()

        return ind

    @staticmethod
    def generate_signals(
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: Params,
    ) -> SignalFrame:
        close = data["close"]

        n = len(data)
        signal_arr = np.zeros(n, dtype=int)
        size_arr = np.ones(n, dtype=float)

        close_vals = close.values
        zscore_vals = indicators["zscore"].values
        gap_queue_vals = indicators["gap_queue"].values
        atr_vals = indicators["atr"].values
        ma200_vals = indicators["ma200"].values

        in_trade = False
        entry_bar = -999
        stop_price = 0.0

        for i in range(n):
            c = close_vals[i]
            z = zscore_vals[i]
            gq = gap_queue_vals[i]
            atr_v = atr_vals[i]
            m200 = ma200_vals[i]

            # Skip warmup bars where indicators are not yet valid
            if np.isnan(z) or np.isnan(gq) or np.isnan(atr_v) or np.isnan(m200):
                continue

            if in_trade:
                bars_held = i - entry_bar
                # Fixed volatility stop: exit if close drops below entry - k*ATR
                # Also exit after max_hold_bars to enforce 1-2 day horizon
                if c < stop_price or bars_held >= params.max_hold_bars:
                    signal_arr[i] = 0
                    in_trade = False
                else:
                    signal_arr[i] = 1
            elif (
                c > m200                           # bull regime: above 200-day MA
                and gq >= params.queue_capacity    # gap queue has overflowed capacity
                and z < params.zscore_threshold    # price is statistically oversold vs MA
            ):
                signal_arr[i] = 1
                in_trade = True
                entry_bar = i
                atr_safe = atr_v if atr_v > 0 else c * 0.01
                stop_price = c - params.atr_stop_k * atr_safe

        signal = pd.Series(signal_arr, index=data.index)
        size = pd.Series(size_arr, index=data.index)

        # Mandatory one-bar shift: decision at bar N fills at bar N+1
        signal = signal.shift(1).fillna(0).astype(int)

        df = pd.DataFrame({"signal": signal, "size": size}, index=data.index)
        return SignalFrame(data=df, signal_column="signal", size_column="size")
