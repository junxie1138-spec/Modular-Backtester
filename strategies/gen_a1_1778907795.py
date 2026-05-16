from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class FreeFlowTrendParams:
    reg_window: int = 20
    rank_window: int = 60
    rank_threshold: float = 0.8
    hold_bars: int = 2
    size_base: float = 1.0
    size_scale: float = 3.0


class GeneratedStrategy(BaseStrategy[FreeFlowTrendParams]):
    strategy_id = "gen_a1_1778907795"

    @classmethod
    def params_type(cls) -> type[FreeFlowTrendParams]:
        return FreeFlowTrendParams

    @staticmethod
    def warmup_bars(params: FreeFlowTrendParams) -> int:
        return int(max(params.reg_window, 3) + max(params.rank_window, 5) + 5)

    @staticmethod
    def indicators(data: pd.DataFrame, params: FreeFlowTrendParams) -> pd.DataFrame:
        close = data["close"].astype(float)
        n = len(close)
        W = max(int(params.reg_window), 3)
        P = max(int(params.rank_window), 5)

        # Rolling linear regression of close on time, fully vectorised.
        # slope = corr(t, y) * std(y) / std(t);  R^2 = corr^2.
        t = pd.Series(np.arange(n, dtype=float), index=close.index)
        corr = close.rolling(W).corr(t)
        std_y = close.rolling(W).std()
        std_x = float(np.std(np.arange(W, dtype=float), ddof=1))
        if not np.isfinite(std_x) or std_x <= 0.0:
            std_x = 1.0
        slope = corr * std_y / std_x

        safe_close = close.replace(0.0, np.nan)
        slope_norm = slope / safe_close          # slope per bar as a fraction of price
        r2 = corr ** 2                            # trend coherence / fit quality

        # Primitive 1: rolling percentile rank of trend SPEED (slope magnitude).
        slope_strength = slope_norm.abs()
        slope_rank = slope_strength.rolling(P).rank(pct=True)
        # Primitive 2: rolling percentile rank of trend COHERENCE (R^2).
        r2_rank = r2.rolling(P).rank(pct=True)

        direction = np.sign(slope_norm)
        direction = pd.Series(direction, index=close.index).fillna(0.0)

        thr = float(params.rank_threshold)
        # Two-primitive AND: both speed and coherence must rank high.
        entry = (
            (slope_rank >= thr)
            & (r2_rank >= thr)
            & (direction != 0.0)
        ).fillna(False)

        out = pd.DataFrame(index=close.index)
        out["slope_norm"] = slope_norm
        out["r2"] = r2
        out["slope_rank"] = slope_rank
        out["r2_rank"] = r2_rank
        out["direction"] = direction
        out["entry"] = entry.astype(float)
        out["conviction"] = (slope_rank.fillna(0.0) + r2_rank.fillna(0.0)) / 2.0
        return out

    @staticmethod
    def generate_signals(
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: FreeFlowTrendParams,
    ) -> SignalFrame:
        n = len(data)
        direction = indicators["direction"].to_numpy(dtype=float)
        entry = indicators["entry"].to_numpy(dtype=float) > 0.5
        conviction = indicators["conviction"].to_numpy(dtype=float)
        hold = max(int(params.hold_bars), 1)
        thr = float(params.rank_threshold)
        base = float(params.size_base)
        scale = float(params.size_scale)

        raw = np.zeros(n, dtype=int)
        size = np.full(n, base, dtype=float)

        remaining = 0
        cur_dir = 0
        cur_size = base
        for i in range(n):
            if remaining > 0:
                # Inside a fixed-bar hold: keep the position.
                raw[i] = cur_dir
                size[i] = cur_size
                remaining -= 1
            elif entry[i] and direction[i] != 0.0 and np.isfinite(conviction[i]):
                # Fresh entry; arm the fixed-bar exit countdown.
                cur_dir = int(np.sign(direction[i]))
                conv = float(conviction[i]) - thr
                cur_size = base + scale * conv
                if not np.isfinite(cur_size):
                    cur_size = base
                cur_size = float(min(max(cur_size, 0.25), 2.0))
                raw[i] = cur_dir
                size[i] = cur_size
                remaining = hold - 1
            else:
                raw[i] = 0
                size[i] = base

        df = pd.DataFrame(index=data.index)
        df["signal"] = raw
        # MANDATORY one-bar shift: decide on bar N close, fill on N+1.
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        size = np.where(np.isfinite(size) & (size > 0.0), size, base)
        df["size"] = size
        return SignalFrame(data=df, signal_column="signal", size_column="size")
