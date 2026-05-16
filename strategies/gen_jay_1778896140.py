from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class Params:
    range_window: int = 10
    body_window: int = 10
    rank_window: int = 60
    entry_threshold: float = 0.70
    atr_window: int = 14
    atr_mult: float = 2.0


class GeneratedStrategy(BaseStrategy[Params]):
    strategy_id = "gen_jay_1778896140"

    @classmethod
    def params_type(cls):
        return Params

    @staticmethod
    def warmup_bars(params: Params) -> int:
        # smooth_range needs range_window bars; diff(range_window) needs range_window more;
        # then rank_window for the percentile; atr_window for ATR; +5 buffer
        return 2 * params.range_window + params.rank_window + params.atr_window + 5

    @staticmethod
    def indicators(data: pd.DataFrame, params: Params) -> pd.DataFrame:
        ind = pd.DataFrame(index=data.index)

        high = data["high"]
        low = data["low"]
        close = data["close"]
        open_ = data["open"]

        hl_range = high - low

        # Primitive 1: range acceleration (second derivative of smoothed daily range)
        # Captures plastic-deformation yield: range expanding faster than its own recent pace
        smooth_range = hl_range.rolling(params.range_window, min_periods=params.range_window).mean()
        range_accel = smooth_range.diff(params.range_window)

        # Primitive 2: directional body efficiency — signed body fraction of total range
        # Positive = bullish pressure dominates the candle; negative = bearish
        body = close - open_
        body_eff = body / hl_range.where(hl_range > 0, other=np.nan)
        smooth_body_eff = body_eff.rolling(params.body_window, min_periods=params.body_window).mean()

        # Percentile ranks of both primitives (vectorised)
        ind["range_accel_rank"] = range_accel.rolling(
            params.rank_window, min_periods=params.rank_window
        ).rank(pct=True)
        ind["body_eff_rank"] = smooth_body_eff.rolling(
            params.rank_window, min_periods=params.rank_window
        ).rank(pct=True)

        # ATR for trailing stop
        prev_close = close.shift(1)
        tr = pd.concat(
            [high - low, (high - prev_close).abs(), (low - prev_close).abs()],
            axis=1,
        ).max(axis=1)
        ind["atr"] = tr.rolling(params.atr_window, min_periods=params.atr_window).mean()

        return ind

    @staticmethod
    def generate_signals(
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: Params,
    ) -> SignalFrame:
        close = data["close"].to_numpy(dtype=float)
        accel_rank = indicators["range_accel_rank"].to_numpy(dtype=float)
        body_rank = indicators["body_eff_rank"].to_numpy(dtype=float)
        atr = indicators["atr"].to_numpy(dtype=float)

        n = len(close)
        signal = np.zeros(n, dtype=np.int64)
        size = np.ones(n, dtype=float)

        thr = params.entry_threshold
        k = params.atr_mult

        position = 0
        high_water = np.nan
        low_water = np.nan

        for i in range(n):
            if np.isnan(accel_rank[i]) or np.isnan(body_rank[i]) or np.isnan(atr[i]):
                continue

            if position == 1:
                # Ratchet high-water mark upward only
                if close[i] > high_water:
                    high_water = close[i]
                # Exit when close drops k*ATR below high-water mark
                if close[i] < high_water - k * atr[i]:
                    signal[i] = 0
                    position = 0
                    high_water = np.nan
                else:
                    signal[i] = 1

            elif position == -1:
                # Ratchet low-water mark downward only
                if close[i] < low_water:
                    low_water = close[i]
                # Exit when close rises k*ATR above low-water mark
                if close[i] > low_water + k * atr[i]:
                    signal[i] = 0
                    position = 0
                    low_water = np.nan
                else:
                    signal[i] = -1

            else:
                # Flat: check two-primitive AND entry conditions
                # Long: range accelerating (plastic regime) AND bullish body efficiency
                if accel_rank[i] > thr and body_rank[i] > thr:
                    signal[i] = 1
                    position = 1
                    high_water = close[i]
                # Short: range accelerating (plastic regime) AND bearish body efficiency
                elif accel_rank[i] > thr and body_rank[i] < 1.0 - thr:
                    signal[i] = -1
                    position = -1
                    low_water = close[i]

        df = pd.DataFrame({"signal": signal, "size": size}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)

        return SignalFrame(data=df, signal_column="signal", size_column="size")
