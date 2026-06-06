from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class Params:
    ma_len: int = 5
    z_len: int = 4
    ac_len: int = 8
    noise_thr: float = 0.10
    z_band: float = 0.50
    size_scale: float = 0.40


class GeneratedStrategy(BaseStrategy[Params]):
    strategy_id = "gen_a1_1778906092"

    @classmethod
    def params_type(cls) -> type[Params]:
        return Params

    @staticmethod
    def warmup_bars(params: Params) -> int:
        # Two parallel (non-additive) chains:
        #   z-score chain  -> ma_len + z_len
        #   autocorr chain -> ac_len + 1 (pct_change costs one bar)
        # +1 guard bar. Defaults give 10, satisfying the warmup<=10 twist.
        return max(params.ma_len + params.z_len, params.ac_len + 1) + 1

    @staticmethod
    def indicators(data: pd.DataFrame, params: Params) -> pd.DataFrame:
        close = data["close"].astype(float)
        out = pd.DataFrame(index=data.index)

        # --- Primary primitive: distance-from-MA z-score ---
        ma = close.rolling(params.ma_len, min_periods=params.ma_len).mean()
        dist = close - ma
        dmean = dist.rolling(params.z_len, min_periods=params.z_len).mean()
        dstd = dist.rolling(params.z_len, min_periods=params.z_len).std(ddof=0)
        z = (dist - dmean) / dstd.replace(0.0, np.nan)
        out["z"] = z.replace([np.inf, -np.inf], np.nan).fillna(0.0)

        # --- Regime detector: lag-1 autocorrelation of close-to-close returns ---
        r = close.pct_change(fill_method=None)
        r_lag = r.shift(1)
        ac = r.rolling(params.ac_len, min_periods=params.ac_len).corr(r_lag)
        out["ac"] = ac.replace([np.inf, -np.inf], np.nan).fillna(0.0)

        return out

    @staticmethod
    def generate_signals(
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: Params,
    ) -> SignalFrame:
        idx = data.index
        n = len(idx)
        z = indicators["z"].to_numpy(dtype=float)
        ac = indicators["ac"].to_numpy(dtype=float)

        # --- Raw entry condition (the side the autocorrelation regime implies) ---
        raw = np.zeros(n, dtype=int)
        for i in range(n):
            zi = z[i]
            ai = ac[i]
            # signal-to-noise filter: regime indeterminate -> no entry condition
            if abs(ai) <= params.noise_thr:
                continue
            # displacement inside its own noise band -> nothing actionable
            if abs(zi) < params.z_band:
                continue
            if ai > 0.0:
                # persistence regime: ride the displacement
                raw[i] = 1 if zi > 0.0 else -1
            else:
                # mean-reversion regime: fade the displacement
                raw[i] = -1 if zi > 0.0 else 1

        # --- Signal-reversal exit: hold until the entry condition flips side ---
        # A raw value of 0 (noise / inside band) never exits; the position is
        # only changed when an opposite directional entry condition fires.
        pos = 0
        sig = np.zeros(n, dtype=int)
        for i in range(n):
            rt = raw[i]
            if rt != 0 and rt != pos:
                pos = rt
            sig[i] = pos

        # Conviction-scaled size from displacement magnitude beyond the band.
        size = 1.0 + params.size_scale * np.clip(
            np.abs(z) - params.z_band, 0.0, 3.0
        )

        df = pd.DataFrame(index=idx)
        df["signal"] = sig
        df["size"] = size
        # MANDATORY one-bar shift: decide on bar N close, fill on bar N+1.
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")
