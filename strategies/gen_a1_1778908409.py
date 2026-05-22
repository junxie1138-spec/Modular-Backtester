from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class ElasticDrawdownParams:
    peak_window: int = 63
    dd_window: int = 21
    rank_window: int = 252
    enter_pct: float = 0.35
    exit_pct: float = 0.70


class GeneratedStrategy(BaseStrategy[ElasticDrawdownParams]):
    """Momentum via drawdown-depth elasticity.

    The recent worst drawdown depth is percentile-ranked against its own
    trailing distribution. A low rank means drawdowns are currently shallow
    relative to history (an 'elastic', resilient momentum regime); a high
    rank means the trend has taken 'plastic' damage. A hysteresis latch keeps
    the position long while the regime stays elastic and exits only when the
    entry condition flips to plastic.
    """

    strategy_id = "gen_a1_1778908409"

    @classmethod
    def params_type(cls):
        return ElasticDrawdownParams

    @staticmethod
    def warmup_bars(params):
        return int(params.peak_window + params.dd_window + params.rank_window + 5)

    @staticmethod
    def indicators(data, params):
        close = data["close"].astype(float)

        peak_w = max(int(params.peak_window), 2)
        dd_w = max(int(params.dd_window), 2)
        rank_w = max(int(params.rank_window), 10)

        # Drawdown from rolling peak (<= 0), NaN-safe via min_periods.
        peak = close.rolling(peak_w, min_periods=2).max()
        dd = (close / peak - 1.0).clip(upper=0.0)

        # Depth of the worst drawdown over the recent window (>= 0).
        dd_depth = (-dd).rolling(dd_w, min_periods=2).max()

        # Percentile rank of current depth vs its own trailing distribution.
        # This is the 'percentile threshold' twist: no fixed drawdown level.
        min_p = max(int(rank_w * 0.5), 20)
        depth_rank = dd_depth.rolling(rank_w, min_periods=min_p).rank(pct=True)

        out = pd.DataFrame(index=data.index)
        out["dd"] = dd
        out["dd_depth"] = dd_depth
        out["depth_rank"] = depth_rank
        return out

    @staticmethod
    def generate_signals(data, indicators, ctx, params):
        rank = indicators["depth_rank"].to_numpy(dtype=float)
        n = len(rank)

        enter_pct = float(params.enter_pct)
        exit_pct = float(params.exit_pct)
        if exit_pct <= enter_pct:
            exit_pct = enter_pct + 0.05

        # Path-dependent elastic/plastic regime latch. The latch state IS the
        # entry condition; exit fires only when it flips to plastic.
        raw = np.zeros(n, dtype=np.int64)
        in_pos = False
        for i in range(n):
            r = rank[i]
            if not np.isfinite(r):
                in_pos = False
                raw[i] = 0
                continue
            if in_pos:
                if r >= exit_pct:        # elastic -> plastic: entry condition flips
                    in_pos = False
            else:
                if r <= enter_pct:       # shallow drawdowns: elastic momentum regime
                    in_pos = True
            raw[i] = 1 if in_pos else 0

        df = pd.DataFrame(index=data.index)
        df["signal"] = raw
        # MANDATORY one-bar shift: decide on bar N close, fill on bar N+1.
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = 1.0
        return SignalFrame(data=df, signal_column="signal", size_column="size")
