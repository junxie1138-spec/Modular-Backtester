from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class GeneratedParams:
    channel: int = 20
    roc_period: int = 10
    atr_period: int = 14
    atr_mult: float = 2.0
    accel_thresh: float = 0.0
    max_hold: int = 10


class GeneratedStrategy(BaseStrategy[GeneratedParams]):
    strategy_id = "gen_a2_1779149478"

    @classmethod
    def params_type(cls) -> type[GeneratedParams]:
        return GeneratedParams

    @staticmethod
    def warmup_bars(params: GeneratedParams) -> int:
        return int(max(params.channel, params.roc_period, params.atr_period)) + 2

    def indicators(self, data: pd.DataFrame, params: GeneratedParams) -> pd.DataFrame:
        close = data["close"]
        high = data["high"]
        low = data["low"]

        # Donchian ceiling: highest high of the prior `channel` bars (excludes current bar).
        donchian_high = high.shift(1).rolling(params.channel, min_periods=params.channel).max()

        # Rate-of-change acceleration: the bar-over-bar change in ROC (second derivative of price).
        roc = close.pct_change(params.roc_period)
        roc_accel = roc.diff()

        # ATR via true range.
        prev_close = close.shift(1)
        tr = pd.concat(
            [
                high - low,
                (high - prev_close).abs(),
                (low - prev_close).abs(),
            ],
            axis=1,
        ).max(axis=1)
        atr = tr.rolling(params.atr_period, min_periods=params.atr_period).mean()

        out = pd.DataFrame(index=data.index)
        out["donchian_high"] = donchian_high
        out["roc_accel"] = roc_accel
        out["atr"] = atr
        return out

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: GeneratedParams,
    ) -> SignalFrame:
        close = data["close"].to_numpy(dtype=float)
        dch = indicators["donchian_high"].to_numpy(dtype=float)
        accel = indicators["roc_accel"].to_numpy(dtype=float)
        atr = indicators["atr"].to_numpy(dtype=float)
        n = len(close)

        # Per-bar qualification: close clears the channel ceiling AND ROC acceleration is positive.
        qualifies = np.zeros(n, dtype=bool)
        for i in range(n):
            if (
                not np.isnan(dch[i])
                and not np.isnan(accel[i])
                and close[i] > dch[i]
                and accel[i] > params.accel_thresh
            ):
                qualifies[i] = True

        signal = np.zeros(n, dtype=int)
        in_pos = False
        stop = 0.0
        bars_held = 0

        for i in range(n):
            if in_pos:
                bars_held += 1
                # Fixed volatility stop (set once at entry) or holding-horizon cap.
                if close[i] <= stop or bars_held >= params.max_hold:
                    signal[i] = 0
                    in_pos = False
                    bars_held = 0
                else:
                    signal[i] = 1
            else:
                # Two-bar confirmation: this bar and the prior bar both qualify.
                if (
                    i >= 1
                    and qualifies[i]
                    and qualifies[i - 1]
                    and not np.isnan(atr[i])
                    and atr[i] > 0.0
                ):
                    in_pos = True
                    bars_held = 0
                    stop = close[i] - params.atr_mult * atr[i]
                    signal[i] = 1

        df = pd.DataFrame(index=data.index)
        df["signal"] = signal
        df["size"] = 1.0
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")
