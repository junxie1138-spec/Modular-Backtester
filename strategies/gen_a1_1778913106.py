from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class RangeReleaseParams:
    lookback: int = 20
    k: float = 2.0


class GeneratedStrategy(BaseStrategy[RangeReleaseParams]):
    """Range-compression release: a percentile-compressed high-low channel is
    armed; a top-quartile single-bar true-range expansion triggers entry in the
    direction of the breakout bar's body. Fixed (non-trailing) ATR vol-stop.
    """

    strategy_id = "gen_a1_1778913106"

    @classmethod
    def params_type(cls) -> type[RangeReleaseParams]:
        return RangeReleaseParams

    @staticmethod
    def warmup_bars(params: RangeReleaseParams) -> int:
        # width needs lookback bars, width_rank stacks another rolling window
        # on top of it; +2 covers .diff() and .shift(1) usage.
        return 2 * int(params.lookback) + 2

    def indicators(self, data: pd.DataFrame, params: RangeReleaseParams) -> pd.DataFrame:
        L = max(2, int(params.lookback))
        high = data["high"].astype(float)
        low = data["low"].astype(float)
        close = data["close"].astype(float)

        prev_close = close.shift(1)
        true_range = pd.concat(
            [
                (high - low),
                (high - prev_close).abs(),
                (low - prev_close).abs(),
            ],
            axis=1,
        ).max(axis=1)  # skipna=True -> first bar falls back to high-low
        atr = true_range.rolling(L, min_periods=L).mean()

        # Geometric channel width, normalised by price, as a percentile rank.
        hh = high.rolling(L, min_periods=L).max()
        ll = low.rolling(L, min_periods=L).min()
        width = (hh - ll) / close.replace(0.0, np.nan)
        width_rank = width.rolling(L, min_periods=L).rank(pct=True)

        # Single-bar range as its own percentile rank (the release trigger).
        bar_range = (high - low) / close.replace(0.0, np.nan)
        range_rank = bar_range.rolling(L, min_periods=L).rank(pct=True)

        dclose = close.diff()

        out = pd.DataFrame(index=data.index)
        out["atr"] = atr
        out["width_rank"] = width_rank
        out["range_rank"] = range_rank
        out["dclose"] = dclose
        return out

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: RangeReleaseParams,
    ) -> SignalFrame:
        k = float(params.k)

        close = data["close"].to_numpy(dtype=float)
        atr = indicators["atr"].to_numpy(dtype=float)
        width_rank = indicators["width_rank"].to_numpy(dtype=float)
        range_rank = indicators["range_rank"].to_numpy(dtype=float)
        dclose = indicators["dclose"].to_numpy(dtype=float)
        n = len(close)

        raw = np.zeros(n, dtype=int)

        # Fixed thresholds (constants, not tunable params): channel compressed
        # when its width rank is in the bottom quartile; release when the bar's
        # range rank is in the top quartile.
        ARM = 0.25
        RELEASE = 0.75
        MAX_HOLD = 2  # 1-2 day horizon

        position = 0
        entry_price = 0.0
        entry_atr = 0.0
        bars_held = 0

        for i in range(n):
            if position == 0:
                armed = (
                    i >= 1
                    and np.isfinite(width_rank[i - 1])
                    and width_rank[i - 1] < ARM
                )
                expansion = np.isfinite(range_rank[i]) and range_rank[i] > RELEASE
                tradable = (
                    armed
                    and expansion
                    and np.isfinite(atr[i])
                    and atr[i] > 0.0
                    and np.isfinite(dclose[i])
                )
                if tradable and dclose[i] > 0.0:
                    position = 1
                    entry_price = close[i]
                    entry_atr = atr[i]
                    bars_held = 0
                elif tradable and dclose[i] < 0.0:
                    position = -1
                    entry_price = close[i]
                    entry_atr = atr[i]
                    bars_held = 0
                raw[i] = position
            else:
                bars_held += 1
                # Fixed volatility stop: ATR captured at entry, never updated.
                stopped = False
                if position == 1 and close[i] <= entry_price - k * entry_atr:
                    stopped = True
                elif position == -1 and close[i] >= entry_price + k * entry_atr:
                    stopped = True
                if stopped or bars_held >= MAX_HOLD:
                    position = 0
                    entry_price = 0.0
                    entry_atr = 0.0
                    bars_held = 0
                    raw[i] = 0
                else:
                    raw[i] = position

        df = pd.DataFrame(index=data.index)
        # MANDATORY one-bar shift: decide on bar N close, fill on bar N+1.
        df["signal"] = pd.Series(raw, index=data.index).shift(1).fillna(0).astype(int)
        df["size"] = 1.0
        return SignalFrame(data=df, signal_column="signal", size_column="size")
