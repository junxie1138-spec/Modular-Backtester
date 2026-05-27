from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class Params:
    range_window: int = 20
    rank_window: int = 60
    compression_pct: float = 0.20
    breakout_pct: float = 0.70
    fade_pct: float = 0.30
    spike_pct: float = 0.90
    refractory_bars: int = 5
    confirm_bars: int = 2
    base_size: float = 1.0


class GeneratedStrategy(BaseStrategy[Params]):
    strategy_id = "gen_a1_1779653574"

    @classmethod
    def params_type(cls) -> type[Params]:
        return Params

    def warmup_bars(self, params: Params) -> int:
        return int(params.range_window + params.rank_window + 5)

    def indicators(self, data: pd.DataFrame, params: Params) -> pd.DataFrame:
        high = data["high"].astype(float)
        low = data["low"].astype(float)
        close = data["close"].astype(float)

        bar_range = (high - low).astype(float)

        roll_high = high.rolling(params.range_window, min_periods=params.range_window).max()
        roll_low = low.rolling(params.range_window, min_periods=params.range_window).min()
        roll_range = (roll_high - roll_low).astype(float)

        denom = roll_range.where(roll_range > 0.0)
        close_pos = (close - roll_low) / denom

        range_rank = roll_range.rolling(params.rank_window, min_periods=params.rank_window).rank(pct=True)
        bar_range_rank = bar_range.rolling(params.rank_window, min_periods=params.rank_window).rank(pct=True)
        close_pos_rank = close_pos.rolling(params.rank_window, min_periods=params.rank_window).rank(pct=True)

        out = pd.DataFrame(
            {
                "range_rank": range_rank,
                "close_pos_rank": close_pos_rank,
                "bar_range_rank": bar_range_rank,
            },
            index=data.index,
        )
        return out

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: Params,
    ) -> SignalFrame:
        range_rank = indicators["range_rank"].to_numpy(dtype=float)
        close_pos_rank = indicators["close_pos_rank"].to_numpy(dtype=float)
        bar_range_rank = indicators["bar_range_rank"].to_numpy(dtype=float)
        n = len(data)

        signal = np.zeros(n, dtype=np.int64)
        position = 0
        refractory_until = -1
        confirm_bars = max(1, int(params.confirm_bars))

        for i in range(n):
            rr_i = range_rank[i]
            cpr_i = close_pos_rank[i]
            brr_i = bar_range_rank[i]

            if not (np.isfinite(rr_i) and np.isfinite(cpr_i) and np.isfinite(brr_i)):
                signal[i] = position
                continue

            if brr_i >= params.spike_pct:
                refractory_until = i + int(params.refractory_bars)

            long_cond = False
            short_cond = False
            if i >= confirm_bars - 1:
                long_ok = True
                short_ok = True
                for k in range(confirm_bars):
                    j = i - k
                    rrk = range_rank[j]
                    cprk = close_pos_rank[j]
                    if not (np.isfinite(rrk) and np.isfinite(cprk)):
                        long_ok = False
                        short_ok = False
                        break
                    compressed = rrk <= params.compression_pct
                    if not (compressed and cprk >= params.breakout_pct):
                        long_ok = False
                    if not (compressed and cprk <= params.fade_pct):
                        short_ok = False
                long_cond = long_ok
                short_cond = short_ok

            in_refractory = i <= refractory_until

            if position == 0:
                if not in_refractory:
                    if long_cond and not short_cond:
                        position = 1
                    elif short_cond and not long_cond:
                        position = -1
            elif position == 1:
                if short_cond and not long_cond:
                    position = -1
            elif position == -1:
                if long_cond and not short_cond:
                    position = 1

            signal[i] = position

        df = pd.DataFrame(index=data.index)
        df["signal"] = signal
        df["size"] = float(max(params.base_size, 1e-6))

        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)

        return SignalFrame(data=df, signal_column="signal", size_column="size")
