from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class GapRankParams:
    gap_window: int = 60
    rank_lo: float = 0.55
    rank_hi: float = 0.92
    vol_short: int = 5
    vol_long: int = 40
    headway_thresh: float = 0.85
    trend_window: int = 100
    profit_target: float = 0.015
    time_stop: int = 2
    base_size: float = 0.5
    size_span: float = 0.7
    max_size: float = 1.5


class GeneratedStrategy(BaseStrategy[GapRankParams]):
    strategy_id = "gen_a1_1778905401"

    @classmethod
    def params_type(cls) -> type[GapRankParams]:
        return GapRankParams

    @staticmethod
    def warmup_bars(params: GapRankParams) -> int:
        return int(max(params.gap_window, params.vol_long, params.trend_window)) + 1

    @staticmethod
    def indicators(data: pd.DataFrame, params: GapRankParams) -> pd.DataFrame:
        close = data["close"]
        open_ = data["open"]

        prev_close = close.shift(1)
        gap = open_ / prev_close - 1.0
        abs_gap = gap.abs()

        # Rolling percentile rank of today's gap magnitude within its own
        # recent distribution -- the primary signal primitive.
        gap_rank = abs_gap.rolling(params.gap_window).rank(pct=True)

        # Traffic-headway gate: ratio of recent short-window gap volatility
        # to longer-window gap volatility. Low ratio == compressed, orderly
        # flow of overnight gaps preceding the current bar.
        vol_short = gap.rolling(params.vol_short).std()
        vol_long = gap.rolling(params.vol_long).std()
        headway = vol_short / vol_long.replace(0.0, np.nan)

        sma = close.rolling(params.trend_window).mean()

        ind = pd.DataFrame(index=data.index)
        ind["gap"] = gap
        ind["gap_rank"] = gap_rank
        ind["headway"] = headway
        ind["sma"] = sma
        return ind

    @staticmethod
    def generate_signals(
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: GapRankParams,
    ) -> SignalFrame:
        close = data["close"].to_numpy(dtype=float)
        gap = indicators["gap"].to_numpy(dtype=float)
        gap_rank = indicators["gap_rank"].to_numpy(dtype=float)
        headway = indicators["headway"].to_numpy(dtype=float)
        sma = indicators["sma"].to_numpy(dtype=float)

        n = len(close)
        signal = np.zeros(n, dtype=int)
        size = np.ones(n, dtype=float)

        rank_lo = float(params.rank_lo)
        rank_hi = float(params.rank_hi)
        band = max(rank_hi - rank_lo, 1e-9)
        pt = float(params.profit_target)
        time_stop = max(int(params.time_stop), 1)
        base_size = float(params.base_size)
        size_span = float(params.size_span)
        max_size = float(params.max_size)
        headway_thresh = float(params.headway_thresh)

        in_pos = False
        entry_price = 0.0
        held = 0
        pos_size = 1.0

        for i in range(n):
            if in_pos:
                held += 1
                hit_target = close[i] >= entry_price * (1.0 + pt)
                hit_time = held >= time_stop
                if hit_target or hit_time:
                    in_pos = False
                    held = 0
                else:
                    signal[i] = 1
                    size[i] = pos_size

            if not in_pos:
                gr = gap_rank[i]
                valid = (
                    np.isfinite(gr)
                    and np.isfinite(gap[i])
                    and np.isfinite(headway[i])
                    and np.isfinite(sma[i])
                )
                if valid:
                    enter = (
                        gap[i] < 0.0
                        and (rank_lo <= gr <= rank_hi)
                        and headway[i] < headway_thresh
                        and close[i] > sma[i]
                    )
                    if enter:
                        frac = (gr - rank_lo) / band
                        if frac < 0.0:
                            frac = 0.0
                        elif frac > 1.0:
                            frac = 1.0
                        sz = base_size + size_span * frac
                        if sz > max_size:
                            sz = max_size
                        if sz < base_size:
                            sz = base_size
                        in_pos = True
                        entry_price = close[i]
                        held = 0
                        pos_size = sz
                        signal[i] = 1
                        size[i] = sz

        df = pd.DataFrame(index=data.index)
        df["signal"] = (
            pd.Series(signal, index=data.index).shift(1).fillna(0).astype(int)
        )
        df["size"] = (
            pd.Series(size, index=data.index).shift(1).fillna(1.0)
        )
        df["size"] = df["size"].clip(lower=1e-6)
        return SignalFrame(data=df, signal_column="signal", size_column="size")
