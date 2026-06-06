from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class GeneratedParams:
    vol_median_window: int = 5
    concord_window: int = 6
    concord_threshold: int = 4
    trend_ma_window: int = 10
    hold_bars: int = 16
    size_min: float = 0.55


class GeneratedStrategy(BaseStrategy[GeneratedParams]):
    strategy_id = "gen_a1_1778884240"

    @classmethod
    def params_type(cls) -> type[GeneratedParams]:
        return GeneratedParams

    @staticmethod
    def warmup_bars(params: GeneratedParams) -> int:
        # Longest lookback: concordant series valid after vol_median_window bars,
        # then a concord_window rolling sum on top of it. diff() adds one bar.
        concord_lookback = params.vol_median_window + params.concord_window - 1
        return int(max(concord_lookback, params.trend_ma_window, 1))

    @staticmethod
    def indicators(data: pd.DataFrame, params: GeneratedParams) -> pd.DataFrame:
        close = data["close"]
        volume = data["volume"]

        ret = close.diff()
        up_day = ret > 0.0

        vol_median = volume.rolling(
            params.vol_median_window, min_periods=1
        ).median()
        vol_strong = volume > vol_median

        # A bar is concordant when price advanced on above-median volume.
        concordant = (up_day & vol_strong).astype(float)
        concord_count = concordant.rolling(
            params.concord_window, min_periods=1
        ).sum()

        ma = close.rolling(params.trend_ma_window, min_periods=1).mean()
        above_ma = (close > ma).astype(float)

        out = pd.DataFrame(index=data.index)
        out["concord_count"] = concord_count.fillna(0.0)
        out["above_ma"] = above_ma.fillna(0.0)
        return out

    @staticmethod
    def generate_signals(
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: GeneratedParams,
    ) -> SignalFrame:
        n = len(data)
        signal = np.zeros(n, dtype=int)
        size = np.ones(n, dtype=float)

        concord_count = indicators["concord_count"].to_numpy(dtype=float)
        above_ma = indicators["above_ma"].to_numpy(dtype=float)

        threshold = float(params.concord_threshold)
        window = float(max(params.concord_window, 1))
        hold = int(max(params.hold_bars, 1))
        size_min = float(min(max(params.size_min, 0.05), 1.0))

        entry = (concord_count >= threshold) & (above_ma > 0.5)

        in_pos = False
        exit_idx = -1
        entry_size = 1.0

        for i in range(n):
            if in_pos and i >= exit_idx:
                # Fixed-bar exit: position closes exactly hold bars after entry.
                in_pos = False

            if not in_pos:
                if entry[i]:
                    in_pos = True
                    exit_idx = i + hold
                    # Tide height: stronger concordance -> larger size.
                    frac = min(max(concord_count[i] / window, 0.0), 1.0)
                    entry_size = size_min + (1.0 - size_min) * frac
                    signal[i] = 1
                    size[i] = entry_size
                else:
                    signal[i] = 0
                    size[i] = 1.0
            else:
                signal[i] = 1
                size[i] = entry_size

        df = data.copy()
        df["signal"] = (
            pd.Series(signal, index=data.index).shift(1).fillna(0).astype(int)
        )
        size_series = pd.Series(size, index=data.index).shift(1).fillna(1.0)
        df["size"] = size_series.clip(lower=0.05).astype(float)

        return SignalFrame(data=df, signal_column="signal", size_column="size")
