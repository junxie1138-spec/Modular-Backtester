from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class GapTensionParams:
    range_win: int = 20
    trend_len: int = 100
    range_compress: float = 0.85
    min_gap_pct: float = 0.0015
    recovery_thresh: float = 0.5
    profit_target: float = 0.03
    max_hold: int = 4
    gap_scale: float = 0.01
    compress_scale: float = 0.4
    size_gain: float = 0.9
    base_size: float = 0.6
    max_size: float = 1.5


class GeneratedStrategy(BaseStrategy[GapTensionParams]):
    strategy_id = "gen_a2_1779146766"

    @classmethod
    def params_type(cls):
        return GapTensionParams

    @staticmethod
    def warmup_bars(params: GapTensionParams) -> int:
        return int(max(params.range_win, params.trend_len)) + 1

    @staticmethod
    def indicators(data: pd.DataFrame, params: GapTensionParams) -> pd.DataFrame:
        out = pd.DataFrame(index=data.index)

        open_ = data["open"]
        high = data["high"]
        low = data["low"]
        close = data["close"]

        prior_close = close.shift(1)
        prior_high = high.shift(1)
        prior_low = low.shift(1)

        bar_range = (high - low).replace(0.0, np.nan)
        range_ma = bar_range.rolling(params.range_win, min_periods=params.range_win).mean()

        gap_pct = open_ / prior_close - 1.0

        # Down-gap contained strictly inside the prior bar's high-low range.
        contained = (open_ < prior_close) & (open_ > prior_low) & (open_ < prior_high)
        deep_enough = gap_pct <= -params.min_gap_pct
        compressed = bar_range < (range_ma * params.range_compress)

        intraday_recovery = (close - low) / bar_range
        recovered = intraday_recovery >= params.recovery_thresh

        sma_trend = close.rolling(params.trend_len, min_periods=params.trend_len).mean()
        uptrend = close > sma_trend

        entry_ok = contained & deep_enough & compressed & recovered & uptrend
        out["entry_ok"] = entry_ok.fillna(False).astype(bool)

        # Elastic-tension components, each clipped to [0, 1].
        gap_depth = np.clip((-gap_pct) / params.gap_scale, 0.0, 1.0)
        compress_amt = np.clip(
            ((range_ma / bar_range) - 1.0) / params.compress_scale, 0.0, 1.0
        )
        recovery_amt = np.clip(intraday_recovery, 0.0, 1.0)

        strength = (gap_depth + compress_amt + recovery_amt) / 3.0
        out["strength"] = strength.fillna(0.0)

        size_target = params.base_size + out["strength"] * params.size_gain
        out["size_target"] = size_target.clip(lower=params.base_size, upper=params.max_size)

        return out

    @staticmethod
    def generate_signals(
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: GapTensionParams,
    ) -> SignalFrame:
        n = len(data)
        close = data["close"].to_numpy(dtype=float)
        entry_ok = indicators["entry_ok"].to_numpy(dtype=bool)
        size_target = indicators["size_target"].to_numpy(dtype=float)

        position = np.zeros(n, dtype=float)
        size = np.ones(n, dtype=float)

        in_pos = False
        entry_price = 0.0
        entry_size = 1.0
        bars_held = 0
        pt = float(params.profit_target)
        max_hold = int(params.max_hold)

        for i in range(n):
            if not in_pos:
                if entry_ok[i]:
                    in_pos = True
                    entry_price = close[i]
                    entry_size = size_target[i]
                    if not np.isfinite(entry_size) or entry_size <= 0.0:
                        entry_size = float(params.base_size)
                    bars_held = 0
                    position[i] = 1.0
                    size[i] = entry_size
                else:
                    position[i] = 0.0
                    size[i] = 1.0
            else:
                bars_held += 1
                gain = close[i] / entry_price - 1.0 if entry_price > 0.0 else 0.0
                if gain >= pt or bars_held >= max_hold:
                    in_pos = False
                    position[i] = 0.0
                    size[i] = 1.0
                else:
                    position[i] = 1.0
                    size[i] = entry_size

        df = pd.DataFrame(index=data.index)
        df["signal"] = pd.Series(position, index=data.index).shift(1).fillna(0).astype(int)
        df["size"] = (
            pd.Series(size, index=data.index).shift(1).fillna(1.0).clip(lower=1e-6)
        )

        return SignalFrame(data=df, signal_column="signal", size_column="size")
