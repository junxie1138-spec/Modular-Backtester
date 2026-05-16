from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class Params:
    peak_window: int = 60
    rank_window: int = 126
    elastic_pct: float = 80.0
    recovery_pct: float = 45.0
    short_calm_pct: float = 20.0
    range_window: int = 21


class GeneratedStrategy(BaseStrategy[Params]):
    strategy_id = "gen_jay_1778916976"

    @classmethod
    def params_type(cls):
        return Params

    @staticmethod
    def warmup_bars(params: Params) -> int:
        return params.rank_window + params.peak_window + 2

    @staticmethod
    def indicators(data: pd.DataFrame, params: Params) -> pd.DataFrame:
        ind = pd.DataFrame(index=data.index)
        close = data["close"]

        peak = close.rolling(params.peak_window).max()
        dd_depth = (peak - close) / peak.replace(0, np.nan)

        dd_vel = dd_depth.diff()
        dd_vel_rank = dd_vel.rolling(params.rank_window).rank(pct=True) * 100.0

        rng_hi = close.rolling(params.range_window).max()
        rng_lo = close.rolling(params.range_window).min()
        price_rank = (
            (close - rng_lo) / (rng_hi - rng_lo).replace(0, np.nan) * 100.0
        )

        stretch_lookback = max(1, params.rank_window // 4)
        was_elastic = (
            (dd_vel_rank >= params.elastic_pct)
            .astype(float)
            .rolling(stretch_lookback)
            .max()
            .fillna(0.0)
        )

        ind["dd_depth"] = dd_depth
        ind["dd_vel_rank"] = dd_vel_rank
        ind["price_rank"] = price_rank
        ind["was_elastic"] = was_elastic

        return ind

    @staticmethod
    def generate_signals(
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: Params,
    ) -> SignalFrame:
        df = pd.DataFrame(index=data.index)
        df["size"] = 1.0

        dd_vel_rank = indicators["dd_vel_rank"]
        price_rank = indicators["price_rank"]
        dd_depth = indicators["dd_depth"]
        was_elastic = indicators["was_elastic"]

        long_cond = (
            (was_elastic > 0)
            & (dd_vel_rank < params.recovery_pct)
            & (dd_depth > 0.005)
        ).fillna(False)

        short_cond = (
            (dd_vel_rank < params.short_calm_pct)
            & (price_rank > 75.0)
            & (dd_depth < 0.015)
        ).fillna(False)

        n = len(df)
        raw = np.zeros(n, dtype=np.int64)
        pos = 0

        lc = long_cond.values
        sc = short_cond.values

        for i in range(n):
            if pos == 0:
                if lc[i]:
                    pos = 1
                elif sc[i]:
                    pos = -1
            elif pos == 1:
                if sc[i]:
                    pos = -1
                elif not lc[i]:
                    pos = 0
            else:
                if lc[i]:
                    pos = 1
                elif not sc[i]:
                    pos = 0
            raw[i] = pos

        raw_series = pd.Series(raw, index=data.index)
        df["signal"] = raw_series.shift(1).fillna(0).astype(int)

        return SignalFrame(data=df, signal_column="signal", size_column="size")
