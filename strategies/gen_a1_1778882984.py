from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class GeneratedParams:
    peak_window: int = 60
    rank_window: int = 252
    overflow_pct: float = 0.90
    confirm: bool = True
    profit_target: float = 0.015
    time_stop: int = 2


class GeneratedStrategy(BaseStrategy[GeneratedParams]):
    strategy_id = "gen_a1_1778882984"

    @classmethod
    def params_type(cls) -> type[GeneratedParams]:
        return GeneratedParams

    @staticmethod
    def warmup_bars(params: GeneratedParams) -> int:
        return int(max(2, params.peak_window) + max(5, params.rank_window) + 2)

    @staticmethod
    def indicators(data: pd.DataFrame, params: GeneratedParams) -> pd.DataFrame:
        close = data["close"]
        peak_window = max(2, int(params.peak_window))
        rank_window = max(5, int(params.rank_window))

        # Drawdown depth from a rolling peak (>= 0, NaN during warmup).
        peak = close.rolling(peak_window, min_periods=peak_window).max()
        safe_peak = peak.replace(0.0, np.nan)
        dd_depth = ((safe_peak - close) / safe_peak).clip(lower=0.0)

        # Rolling percentile rank of drawdown depth = capacity/overflow gauge.
        min_p = max(20, rank_window // 2)
        dd_rank = dd_depth.rolling(rank_window, min_periods=min_p).rank(pct=True)

        ret1 = close.pct_change()

        out = pd.DataFrame(index=data.index)
        out["dd_depth"] = dd_depth
        out["dd_rank"] = dd_rank
        out["ret1"] = ret1
        return out

    @staticmethod
    def generate_signals(
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: GeneratedParams,
    ) -> SignalFrame:
        close = data["close"].to_numpy(dtype=float)
        dd_rank = indicators["dd_rank"].to_numpy(dtype=float)
        ret1 = indicators["ret1"].to_numpy(dtype=float)

        n = len(close)
        overflow = float(params.overflow_pct)
        low_thr = 1.0 - overflow
        confirm = bool(params.confirm)
        target = float(params.profit_target)
        tstop = max(1, int(params.time_stop))

        # Percentile-threshold entries (the twist: rank vs its own distribution).
        long_entry = np.zeros(n, dtype=bool)
        short_entry = np.zeros(n, dtype=bool)
        for i in range(n):
            r = dd_rank[i]
            if not np.isfinite(r):
                continue
            m = ret1[i]
            if r >= overflow:
                if (not confirm) or (np.isfinite(m) and m > 0.0):
                    long_entry[i] = True
            elif r <= low_thr:
                if (not confirm) or (np.isfinite(m) and m < 0.0):
                    short_entry[i] = True

        # Path-dependent exit: profit-target OR time-stop, whichever first.
        pos = np.zeros(n, dtype=int)
        current = 0
        entry_price = 0.0
        bars_held = 0
        for i in range(n):
            if current == 0:
                if long_entry[i]:
                    current = 1
                    entry_price = close[i]
                    bars_held = 0
                elif short_entry[i]:
                    current = -1
                    entry_price = close[i]
                    bars_held = 0
                pos[i] = current
            else:
                bars_held += 1
                if current == 1 and entry_price > 0.0:
                    gain = close[i] / entry_price - 1.0
                elif current == -1 and close[i] > 0.0:
                    gain = entry_price / close[i] - 1.0
                else:
                    gain = 0.0
                if gain >= target or bars_held >= tstop:
                    current = 0
                pos[i] = current

        df = pd.DataFrame(index=data.index)
        df["signal"] = pos
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = 1.0
        return SignalFrame(data=df, signal_column="signal", size_column="size")
