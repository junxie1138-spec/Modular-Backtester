from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class ShockwaveRegimeParams:
    roc_len: int = 5
    accel_lag: int = 3
    accel_std_len: int = 20
    regime_len: int = 12
    regime_thresh: float = 0.2
    entry_z: float = 1.0
    size_cap: float = 3.0
    base_size: float = 1.0


class GeneratedStrategy(BaseStrategy[ShockwaveRegimeParams]):
    strategy_id = "gen_a2_1779147960"

    @classmethod
    def params_type(cls):
        return ShockwaveRegimeParams

    @staticmethod
    def warmup_bars(params: ShockwaveRegimeParams) -> int:
        return int(
            params.roc_len
            + params.accel_lag
            + params.accel_std_len
            + params.regime_len
            + 5
        )

    def indicators(self, data: pd.DataFrame, params: ShockwaveRegimeParams) -> pd.DataFrame:
        p = params
        close = data["close"]

        # Velocity: rate of change over roc_len bars.
        roc = close.pct_change(p.roc_len)
        # Acceleration: rate-of-change of the rate-of-change (second derivative).
        accel = roc - roc.shift(p.accel_lag)

        accel_std = accel.rolling(p.accel_std_len, min_periods=p.accel_std_len).std()
        accel_std = accel_std.replace(0.0, np.nan)
        accel_z = (accel / accel_std).replace([np.inf, -np.inf], np.nan).fillna(0.0)

        vel_sign = np.sign(roc.fillna(0.0))
        acc_sign = np.sign(accel.fillna(0.0))

        # Traffic metaphor: aligned (+1) = free flow, opposed (-1) = congestion front.
        aligned = vel_sign * acc_sign
        regime_score = (
            aligned.rolling(p.regime_len, min_periods=p.regime_len)
            .mean()
            .fillna(0.0)
        )

        ind = pd.DataFrame(index=data.index)
        ind["roc"] = roc.fillna(0.0)
        ind["accel"] = accel.fillna(0.0)
        ind["accel_z"] = accel_z
        ind["vel_sign"] = vel_sign
        ind["regime_score"] = regime_score
        return ind

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: ShockwaveRegimeParams,
    ) -> SignalFrame:
        p = params
        ind = indicators
        accel_z = ind["accel_z"]
        vel_sign = ind["vel_sign"]
        regime_score = ind["regime_score"]

        momentum_regime = regime_score > p.regime_thresh
        reversal_regime = regime_score < -p.regime_thresh

        # Free-flow regime trades with velocity; congestion-front regime fades it.
        raw_dir = pd.Series(0.0, index=data.index)
        raw_dir = raw_dir.where(~momentum_regime, vel_sign)
        raw_dir = raw_dir.where(~reversal_regime, -vel_sign)

        # Only act when the acceleration shock is large enough.
        strong = accel_z.abs() > p.entry_z
        raw_dir = raw_dir.where(strong, 0.0).fillna(0.0)

        # Two-bar confirmation: same proposed direction on two consecutive bars.
        prev = raw_dir.shift(1).fillna(0.0)
        confirmed = raw_dir.where((raw_dir == prev) & (raw_dir != 0.0), 0.0)
        cand = confirmed.to_numpy()

        # Signal-reversal exit: hold until the opposite confirmed entry fires.
        n = len(cand)
        sig = np.zeros(n, dtype=int)
        pos = 0
        for i in range(n):
            c = int(cand[i])
            if pos == 0:
                if c != 0:
                    pos = c
            else:
                if c == -pos:
                    pos = c
            sig[i] = pos

        size_cap = p.size_cap if p.size_cap > 0.0 else 1.0
        size = p.base_size * (0.5 + np.minimum(accel_z.abs(), size_cap) / size_cap)
        size = size.clip(lower=0.01).fillna(p.base_size)

        df = pd.DataFrame(index=data.index)
        df["signal"] = sig
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = size.to_numpy()
        return SignalFrame(data=df, signal_column="signal", size_column="size")
