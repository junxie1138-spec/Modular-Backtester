from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class PredatorPreyGapParams:
    gap_lookback: int = 60
    gap_percentile: float = 5.0
    predator_window: int = 10
    predator_percentile: float = 90.0
    profit_target_pct: float = 0.015
    time_stop_bars: int = 4
    trend_filter_period: int = 200


class GeneratedStrategy(BaseStrategy[PredatorPreyGapParams]):
    strategy_id = "gen_a1_1779877749"

    @classmethod
    def params_type(cls):
        return PredatorPreyGapParams

    @classmethod
    def warmup_bars(cls, params: PredatorPreyGapParams) -> int:
        return int(max(params.trend_filter_period, params.gap_lookback + params.predator_window) + 5)

    def indicators(self, data: pd.DataFrame, params: PredatorPreyGapParams) -> pd.DataFrame:
        out = pd.DataFrame(index=data.index)

        gap_lb = int(params.gap_lookback)
        pred_w = int(params.predator_window)
        trend_p = int(params.trend_filter_period)

        prev_close = data["close"].shift(1)
        gap = (data["open"] - prev_close) / prev_close.replace(0.0, np.nan)
        out["gap"] = gap

        # Rolling percentile rank (0-100) of today's gap within the trailing window.
        # A low rank (<= gap_percentile) means today's gap is in the extreme negative tail.
        gap_rank = gap.rolling(gap_lb, min_periods=gap_lb).rank(pct=True) * 100.0
        out["gap_pct_rank"] = gap_rank

        # Predator population: cumulative magnitude of negative gaps over predator_window.
        neg_gap_mag = (-gap).clip(lower=0.0).fillna(0.0)
        predator_pop = neg_gap_mag.rolling(pred_w, min_periods=pred_w).sum()
        out["predator_pop"] = predator_pop

        # Percentile rank of the predator population itself - saturation tail.
        predator_rank = predator_pop.rolling(gap_lb, min_periods=gap_lb).rank(pct=True) * 100.0
        out["predator_pct_rank"] = predator_rank

        # Long-only trend filter.
        out["sma_long"] = data["close"].rolling(trend_p, min_periods=trend_p).mean()

        return out

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: PredatorPreyGapParams,
    ) -> SignalFrame:
        close = data["close"].to_numpy(dtype=float)
        gap_rank = indicators["gap_pct_rank"].to_numpy(dtype=float)
        pred_rank = indicators["predator_pct_rank"].to_numpy(dtype=float)
        sma = indicators["sma_long"].to_numpy(dtype=float)

        n = len(close)
        raw = np.zeros(n, dtype=np.int64)

        gp = float(params.gap_percentile)
        pp = float(params.predator_percentile)
        pt = float(params.profit_target_pct)
        ts = int(params.time_stop_bars)

        in_pos = False
        entry_px = np.nan
        bars_held = 0

        for i in range(n):
            if in_pos:
                bars_held += 1
                c = close[i]
                exit_now = False
                if not np.isnan(c) and not np.isnan(entry_px):
                    if c >= entry_px * (1.0 + pt):
                        exit_now = True
                if bars_held >= ts:
                    exit_now = True
                if exit_now:
                    in_pos = False
                    entry_px = np.nan
                    bars_held = 0
                    raw[i] = 0
                else:
                    raw[i] = 1
            else:
                gr = gap_rank[i]
                pr = pred_rank[i]
                sm = sma[i]
                c = close[i]
                if (
                    not np.isnan(gr)
                    and not np.isnan(pr)
                    and not np.isnan(sm)
                    and not np.isnan(c)
                    and gr <= gp
                    and pr >= pp
                    and c > sm
                ):
                    in_pos = True
                    entry_px = c
                    bars_held = 0
                    raw[i] = 1
                else:
                    raw[i] = 0

        df = pd.DataFrame(index=data.index)
        df["signal"] = raw
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = 1.0
        return SignalFrame(data=df, signal_column="signal", size_column="size")
