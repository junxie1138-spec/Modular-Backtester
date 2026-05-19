from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class GeneratedParams:
    win_window: int = 15
    rank_window: int = 120
    entry_pct: float = 0.80
    atr_window: int = 14
    atr_k: float = 2.5
    max_hold: int = 5


class GeneratedStrategy(BaseStrategy[GeneratedParams]):
    strategy_id = "gen_a2_1779152543"

    @classmethod
    def params_type(cls):
        return GeneratedParams

    @staticmethod
    def warmup_bars(params: GeneratedParams) -> int:
        rank_lb = int(params.win_window) + int(params.rank_window) + 2
        atr_lb = int(params.atr_window) + 1
        return int(max(rank_lb, atr_lb))

    @staticmethod
    def indicators(data: pd.DataFrame, params: GeneratedParams) -> pd.DataFrame:
        out = pd.DataFrame(index=data.index)

        close = data["close"].astype(float)
        high = data["high"].astype(float)
        low = data["low"].astype(float)

        win_window = max(int(params.win_window), 2)
        rank_window = max(int(params.rank_window), 5)
        atr_window = max(int(params.atr_window), 2)

        # Fraction of up-closes over the recent window (directional hit-rate).
        up = (close.diff() > 0).astype(float)
        up[close.diff().isna()] = np.nan
        hit_rate = up.rolling(win_window, min_periods=win_window).mean()

        # Rolling percentile rank of the hit-rate within its own history.
        win_rank = hit_rate.rolling(rank_window, min_periods=rank_window).rank(pct=True)
        out["win_rank"] = win_rank

        # Average true range for the fixed volatility stop.
        prev_close = close.shift(1)
        tr = pd.concat(
            [
                (high - low).abs(),
                (high - prev_close).abs(),
                (low - prev_close).abs(),
            ],
            axis=1,
        ).max(axis=1)
        out["atr"] = tr.rolling(atr_window, min_periods=atr_window).mean()

        return out

    @staticmethod
    def generate_signals(
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: GeneratedParams,
    ) -> SignalFrame:
        n = len(data)
        close = data["close"].astype(float).to_numpy()
        rank = indicators["win_rank"].to_numpy()
        atr = indicators["atr"].to_numpy()

        entry_pct = float(params.entry_pct)
        atr_k = float(params.atr_k)
        max_hold = max(int(params.max_hold), 1)

        sig = np.zeros(n, dtype=int)

        in_pos = False
        stop_level = 0.0
        bars_held = 0

        for i in range(n):
            if not in_pos:
                if (
                    i >= 1
                    and np.isfinite(rank[i])
                    and np.isfinite(rank[i - 1])
                    and np.isfinite(atr[i])
                    and atr[i] > 0.0
                ):
                    # Two-bar confirmation: top-band hit-rate rank on both bars.
                    if rank[i] >= entry_pct and rank[i - 1] >= entry_pct:
                        in_pos = True
                        stop_level = close[i] - atr_k * atr[i]
                        bars_held = 0
                        sig[i] = 1
            else:
                bars_held += 1
                # Fixed volatility stop (from entry close) or horizon cap.
                if close[i] <= stop_level or bars_held >= max_hold:
                    in_pos = False
                    sig[i] = 0
                else:
                    sig[i] = 1

        df = pd.DataFrame(index=data.index)
        df["signal"] = pd.Series(sig, index=data.index).shift(1).fillna(0).astype(int)
        df["size"] = 1.0

        return SignalFrame(data=df, signal_column="signal", size_column="size")
