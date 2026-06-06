from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class GapClusterParams:
    gap_count_window: int = 10
    rank_window: int = 120
    upper_pct: float = 0.80
    lower_pct: float = 0.40
    atr_window: int = 14
    atr_mult: float = 2.5
    min_hold_bars: int = 3
    max_hold_bars: int = 10
    trend_window: int = 200
    use_trend_filter: bool = True
    gap_threshold: float = 0.0


class GeneratedStrategy(BaseStrategy[GapClusterParams]):
    strategy_id = "gen_a1_1778906231"

    @classmethod
    def params_type(cls) -> type[GapClusterParams]:
        return GapClusterParams

    def warmup_bars(self, params: GapClusterParams) -> int:
        return int(
            max(
                params.rank_window + params.gap_count_window + 1,
                params.atr_window + 1,
                params.trend_window,
            )
        ) + 5

    def indicators(self, data: pd.DataFrame, params: GapClusterParams) -> pd.DataFrame:
        close = data["close"]
        high = data["high"]
        low = data["low"]
        open_ = data["open"]

        prev_close = close.shift(1)
        gap = open_ / prev_close - 1.0

        up_gap = (gap > params.gap_threshold).astype(float)
        gap_count = up_gap.rolling(
            params.gap_count_window, min_periods=params.gap_count_window
        ).sum()
        gap_rank = gap_count.rolling(
            params.rank_window, min_periods=params.rank_window
        ).rank(pct=True)

        tr = pd.concat(
            [
                (high - low),
                (high - prev_close).abs(),
                (low - prev_close).abs(),
            ],
            axis=1,
        ).max(axis=1)
        atr = tr.rolling(params.atr_window, min_periods=params.atr_window).mean()

        sma = close.rolling(
            params.trend_window, min_periods=params.trend_window
        ).mean()

        out = pd.DataFrame(index=data.index)
        out["gap"] = gap
        out["up_gap"] = up_gap
        out["gap_count"] = gap_count
        out["gap_rank"] = gap_rank
        out["atr"] = atr
        out["sma"] = sma
        return out

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: GapClusterParams,
    ) -> SignalFrame:
        n = len(data)
        close = data["close"].to_numpy(dtype=float)
        gap = indicators["gap"].to_numpy(dtype=float)
        gap_rank = indicators["gap_rank"].to_numpy(dtype=float)
        atr = indicators["atr"].to_numpy(dtype=float)
        sma = indicators["sma"].to_numpy(dtype=float)

        # Schmitt-trigger hysteresis latch on gap-occurrence frequency rank.
        regime = np.zeros(n, dtype=bool)
        on = False
        for i in range(n):
            r = gap_rank[i]
            if not np.isnan(r):
                if (not on) and r >= params.upper_pct:
                    on = True
                elif on and r <= params.lower_pct:
                    on = False
            regime[i] = on

        sig = np.zeros(n, dtype=np.int64)
        in_pos = False
        high_water = 0.0
        bars_held = 0

        for i in range(n):
            c = close[i]
            a = atr[i]

            if in_pos:
                bars_held += 1
                if c > high_water:
                    high_water = c

                exit_now = False
                if bars_held >= params.min_hold_bars:
                    if (not np.isnan(a)) and c <= high_water - params.atr_mult * a:
                        exit_now = True
                    if not regime[i]:
                        exit_now = True
                    if bars_held >= params.max_hold_bars:
                        exit_now = True

                if exit_now:
                    in_pos = False
                    high_water = 0.0
                    bars_held = 0
                    sig[i] = 0
                else:
                    sig[i] = 1
            else:
                enter = (
                    regime[i]
                    and (not np.isnan(gap[i]))
                    and gap[i] > params.gap_threshold
                    and (not np.isnan(a))
                )
                if enter and params.use_trend_filter:
                    enter = (not np.isnan(sma[i])) and (c > sma[i])

                if enter:
                    in_pos = True
                    high_water = c
                    bars_held = 0
                    sig[i] = 1
                else:
                    sig[i] = 0

        df = pd.DataFrame(index=data.index)
        df["signal"] = sig
        df["size"] = 1.0
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")
