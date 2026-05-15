from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class PredatorPreyParams:
    vol_window: int = 20
    alpha: float = 0.60
    beta: float = 0.05
    gamma: float = 0.04
    delta: float = 0.25
    p_init: float = 1.0
    q_init: float = 1.0
    pop_cap: float = 50.0
    burn_in: int = 60
    base_size: float = 0.40
    size_scale: float = 0.80
    size_min: float = 0.20
    size_max: float = 1.00


class GeneratedStrategy(BaseStrategy[PredatorPreyParams]):
    strategy_id = "gen_a1_1778883238"

    @classmethod
    def params_type(cls) -> type[PredatorPreyParams]:
        return PredatorPreyParams

    @staticmethod
    def warmup_bars(params: PredatorPreyParams) -> int:
        return int(params.vol_window) + int(params.burn_in) + 1

    @staticmethod
    def indicators(data: pd.DataFrame, params: PredatorPreyParams) -> pd.DataFrame:
        close = data["close"].astype(float)
        ret = close.pct_change()

        win = max(int(params.vol_window), 2)
        vol = ret.rolling(win, min_periods=win).std()
        ret_norm = ret / vol.replace(0.0, np.nan)
        ret_norm = ret_norm.replace([np.inf, -np.inf], np.nan)

        food = ret_norm.to_numpy(dtype=float)
        n = len(food)
        prey = np.zeros(n, dtype=float)
        pred = np.zeros(n, dtype=float)

        p = float(params.p_init)
        q = float(params.q_init)
        cap = float(params.pop_cap)
        alpha = float(params.alpha)
        beta = float(params.beta)
        gamma = float(params.gamma)
        delta = float(params.delta)

        for i in range(n):
            f = food[i]
            if not np.isfinite(f) or f < 0.0:
                f = 0.0
            interaction = beta * p * q
            p_new = p + alpha * f - interaction
            q_new = q + gamma * p * q - delta * q
            if not np.isfinite(p_new):
                p_new = 0.0
            if not np.isfinite(q_new):
                q_new = 0.0
            p = min(max(p_new, 0.0), cap)
            q = min(max(q_new, 0.0), cap)
            prey[i] = p
            pred[i] = q

        out = pd.DataFrame(index=data.index)
        out["ret"] = ret
        out["ret_norm"] = ret_norm
        out["prey"] = prey
        out["predator"] = pred
        denom = prey + pred
        gap = np.where(denom > 0.0, (prey - pred) / denom, 0.0)
        out["gap"] = pd.Series(gap, index=data.index)
        return out

    @staticmethod
    def generate_signals(
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: PredatorPreyParams,
    ) -> SignalFrame:
        prey = indicators["prey"].astype(float)
        pred = indicators["predator"].astype(float)
        gap = indicators["gap"].astype(float).fillna(0.0)

        # Entry condition: prey (momentum) population dominates predator.
        # Signal-reversal exit: holding continues until prey > predator flips
        # false, at which point the raw signal becomes 0 on its own.
        raw = (prey > pred).astype(int)

        warmup = GeneratedStrategy.warmup_bars(params)
        if warmup > 0:
            raw.iloc[:min(warmup, len(raw))] = 0

        # Signal-scaled position sizing: larger when the prey-predator
        # dominance gap is wider (more robust momentum regime).
        size = params.base_size + params.size_scale * gap.clip(lower=0.0)
        size = size.clip(lower=params.size_min, upper=params.size_max)

        df = pd.DataFrame(index=data.index)
        df["signal"] = raw.astype(int)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)

        df["size"] = size.shift(1).fillna(float(params.base_size))
        df["size"] = df["size"].clip(
            lower=params.size_min, upper=params.size_max
        ).astype(float)

        return SignalFrame(data=df, signal_column="signal", size_column="size")
