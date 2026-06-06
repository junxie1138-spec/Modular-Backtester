from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy

# Fixed (non-tunable) mechanism constants - the twist caps tunable params at 2.
_GAP_WINDOW = 20      # rolling window for gap mean/std (the spring's "natural length")
_TIME_STOP = 10       # bars held before forced exit (~2 trading weeks)
_MAX_SIZE = 2.0       # cap on compression-scaled position size


@dataclass(slots=True)
class GapSpringParams:
    # Tunable param 1: how many std devs below the gap's normal range triggers a buy.
    gap_z: float = 1.5
    # Tunable param 2: profit target as a fraction of entry close price.
    profit_target: float = 0.03


class GeneratedStrategy(BaseStrategy[GapSpringParams]):
    strategy_id = "gen_a1_1778884473"

    @classmethod
    def params_type(cls) -> type[GapSpringParams]:
        return GapSpringParams

    @staticmethod
    def warmup_bars(params: GapSpringParams) -> int:
        # gap uses close.shift(1); gap_z uses a rolling window of length _GAP_WINDOW.
        # _GAP_WINDOW + 2 covers the diff plus the rolling window with margin.
        return _GAP_WINDOW + 2

    @staticmethod
    def indicators(data: pd.DataFrame, params: GapSpringParams) -> pd.DataFrame:
        close = data["close"].astype(float)
        open_ = data["open"].astype(float)

        prior_close = close.shift(1)
        # Overnight gap: the spring displacement from prior settlement.
        gap = open_ / prior_close - 1.0

        gmean = gap.rolling(_GAP_WINDOW).mean()
        gstd = gap.rolling(_GAP_WINDOW).std()
        # Standardized gap: how stretched the spring is vs its recent natural range.
        gap_z = (gap - gmean) / gstd.replace(0.0, np.nan)

        out = pd.DataFrame(index=data.index)
        out["gap"] = gap
        out["gap_z"] = gap_z
        return out

    @staticmethod
    def generate_signals(
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: GapSpringParams,
    ) -> SignalFrame:
        n = len(data)
        close = data["close"].to_numpy(dtype=float)
        gap_z = indicators["gap_z"].to_numpy(dtype=float)

        raw = np.zeros(n, dtype=np.int64)
        size_arr = np.ones(n, dtype=float)

        thresh = float(params.gap_z)
        pt = float(params.profit_target)
        if thresh <= 0.0:
            thresh = 1e-9  # guard against degenerate optimizer values

        in_pos = False
        entry_price = 0.0
        bars_held = 0
        current_size = 1.0

        # Path-dependent profit-target + time-stop exit requires a bar-indexed loop.
        for i in range(n):
            if in_pos:
                bars_held += 1
                hit_pt = close[i] >= entry_price * (1.0 + pt)
                hit_ts = bars_held >= _TIME_STOP
                if hit_pt or hit_ts:
                    in_pos = False
                    raw[i] = 0
                    bars_held = 0
                else:
                    raw[i] = 1
                    size_arr[i] = current_size
            else:
                gz = gap_z[i]
                # Extreme down-gap: spring compressed well below its natural range.
                if np.isfinite(gz) and gz <= -thresh:
                    in_pos = True
                    entry_price = close[i]
                    bars_held = 0
                    # Deeper compression -> stronger restoring force -> larger size.
                    current_size = float(min(_MAX_SIZE, max(1.0, abs(gz) / thresh)))
                    raw[i] = 1
                    size_arr[i] = current_size

        df = pd.DataFrame(index=data.index)
        df["signal"] = raw
        df["size"] = size_arr

        # MANDATORY one-bar shift: decision on bar N's close fills on bar N+1.
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = df["size"].shift(1).fillna(1.0).astype(float)

        return SignalFrame(data=df, signal_column="signal", size_column="size")
