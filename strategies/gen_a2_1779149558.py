from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class GeneratedParams:
    fast_range_len: int = 5
    slow_range_len: int = 40
    expand_thresh: float = 1.10
    trend_len: int = 100
    clv_thresh: float = 0.55
    atr_len: int = 14
    atr_stop_mult: float = 2.5
    max_hold_bars: int = 18


class GeneratedStrategy(BaseStrategy[GeneratedParams]):
    strategy_id = "gen_a2_1779149558"

    @classmethod
    def params_type(cls):
        return GeneratedParams

    def warmup_bars(self, params: GeneratedParams) -> int:
        return int(max(params.slow_range_len, params.trend_len, params.atr_len + 1)) + 3

    def indicators(self, data: pd.DataFrame, params: GeneratedParams) -> pd.DataFrame:
        high = data["high"].astype(float)
        low = data["low"].astype(float)
        close = data["close"].astype(float)
        prev_close = close.shift(1)

        rng = (high - low).clip(lower=0.0)

        tr = pd.concat(
            [
                (high - low).abs(),
                (high - prev_close).abs(),
                (low - prev_close).abs(),
            ],
            axis=1,
        ).max(axis=1)
        atr = tr.rolling(params.atr_len, min_periods=params.atr_len).mean()

        fast_range = rng.rolling(params.fast_range_len, min_periods=params.fast_range_len).mean()
        slow_range = rng.rolling(params.slow_range_len, min_periods=params.slow_range_len).mean()
        slow_safe = slow_range.where(slow_range > 0.0)
        range_ratio = fast_range / slow_safe

        expanding = (range_ratio > params.expand_thresh).fillna(False)

        ma = close.rolling(params.trend_len, min_periods=params.trend_len).mean()
        above_ma = (close > ma).fillna(False)

        rng_arr = rng.to_numpy(dtype=float)
        clv_arr = np.where(
            rng_arr > 0.0,
            (close.to_numpy(dtype=float) - low.to_numpy(dtype=float)) / np.where(rng_arr > 0.0, rng_arr, 1.0),
            0.5,
        )
        clv = pd.Series(clv_arr, index=data.index)
        upper_close = (clv > params.clv_thresh).fillna(False)

        out = pd.DataFrame(index=data.index)
        out["expanding"] = expanding.astype(float)
        out["above_ma"] = above_ma.astype(float)
        out["upper_close"] = upper_close.astype(float)
        out["atr"] = atr.astype(float)
        return out

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: GeneratedParams,
    ) -> SignalFrame:
        close = data["close"].to_numpy(dtype=float)
        n = len(close)

        expanding = indicators["expanding"].to_numpy(dtype=float) > 0.5
        above_ma = indicators["above_ma"].to_numpy(dtype=float) > 0.5
        upper_close = indicators["upper_close"].to_numpy(dtype=float) > 0.5
        atr = indicators["atr"].to_numpy(dtype=float)

        signal = np.zeros(n, dtype=np.int64)
        warm = self.warmup_bars(params)

        in_pos = False
        stop_level = 0.0
        bars_held = 0

        for i in range(n):
            if i < warm or i < 2:
                continue

            if in_pos:
                bars_held += 1
                exit_now = close[i] < stop_level or bars_held >= params.max_hold_bars
                if exit_now:
                    in_pos = False
                    bars_held = 0
                    stop_level = 0.0
                    signal[i] = 0
                else:
                    signal[i] = 1
                continue

            confirm = expanding[i] and expanding[i - 1]
            transition = not expanding[i - 2]
            direction = upper_close[i] and upper_close[i - 1]
            trend = above_ma[i]
            valid_atr = np.isfinite(atr[i]) and atr[i] > 0.0

            if confirm and transition and direction and trend and valid_atr:
                in_pos = True
                bars_held = 0
                stop_level = close[i] - params.atr_stop_mult * atr[i]
                signal[i] = 1
            else:
                signal[i] = 0

        df = pd.DataFrame(index=data.index)
        df["signal"] = signal
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = 1.0
        return SignalFrame(data=df, signal_column="signal", size_column="size")
