from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class RangeChannelParams:
    structure_window: int = 8
    enter_thr: float = 0.60
    exit_thr: float = 0.35
    vol_window: int = 20
    target_vol: float = 0.15
    size_min: float = 0.20
    size_max: float = 1.00


class GeneratedStrategy(BaseStrategy[RangeChannelParams]):
    strategy_id = "gen_a1_1778911819"

    @classmethod
    def params_type(cls) -> type[RangeChannelParams]:
        return RangeChannelParams

    @staticmethod
    def warmup_bars(params: RangeChannelParams) -> int:
        n = max(int(params.structure_window), 1)
        m = max(int(params.vol_window), 2)
        return int(max(n + 1, m) + 1)

    @staticmethod
    def indicators(data: pd.DataFrame, params: RangeChannelParams) -> pd.DataFrame:
        high = data["high"].astype(float)
        low = data["low"].astype(float)

        n = max(int(params.structure_window), 1)
        m = max(int(params.vol_window), 2)

        # Primitive 1: range-top dynamics - fraction of recent bars making a higher high.
        higher_high = (high > high.shift(1)).astype(float)
        hh_frac = higher_high.rolling(n).mean()

        # Primitive 2: range-bottom dynamics - fraction of recent bars making a higher low.
        higher_low = (low > low.shift(1)).astype(float)
        hl_frac = higher_low.rolling(n).mean()

        # Parkinson high-low volatility estimator, annualised, for vol targeting.
        ratio = (high / low.replace(0.0, np.nan)).clip(lower=1e-12)
        log_hl = np.log(ratio)
        park_var = (log_hl ** 2).rolling(m).mean() / (4.0 * np.log(2.0))
        park_vol_ann = np.sqrt(park_var.clip(lower=0.0)) * np.sqrt(252.0)
        park_vol_ann = park_vol_ann.replace(0.0, np.nan)

        size_raw = float(params.target_vol) / park_vol_ann
        size_raw = size_raw.clip(lower=float(params.size_min), upper=float(params.size_max))
        size_raw = size_raw.fillna(float(params.size_min))

        out = pd.DataFrame(index=data.index)
        out["hh_frac"] = hh_frac
        out["hl_frac"] = hl_frac
        out["size_raw"] = size_raw
        return out

    @staticmethod
    def generate_signals(
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: RangeChannelParams,
    ) -> SignalFrame:
        idx = data.index
        n_bars = len(idx)

        hh = indicators["hh_frac"].to_numpy(dtype=float)
        hl = indicators["hl_frac"].to_numpy(dtype=float)
        size_raw = indicators["size_raw"].to_numpy(dtype=float)

        enter_thr = float(params.enter_thr)
        exit_thr = float(params.exit_thr)
        # Enforce hysteresis: the exit band must sit strictly below the entry band.
        if exit_thr >= enter_thr:
            exit_thr = enter_thr * 0.5

        valid = ~(np.isnan(hh) | np.isnan(hl))

        # Two-primitive AND: both range boundaries must be trending up to enter.
        enter_cond = valid & (hh >= enter_thr) & (hl >= enter_thr)
        # Signal-reversal exit with hysteresis: the up-channel structure has flipped
        # when either boundary fraction drops back below the lower band.
        exit_cond = valid & ((hh < exit_thr) | (hl < exit_thr))

        raw = np.zeros(n_bars, dtype=int)
        state = 0
        for i in range(n_bars):
            if state == 0:
                if enter_cond[i]:
                    state = 1
            else:
                if exit_cond[i]:
                    state = 0
            raw[i] = state

        size = np.where(np.isnan(size_raw), float(params.size_min), size_raw)
        size = np.clip(size, float(params.size_min), float(params.size_max))

        df = pd.DataFrame(index=idx)
        df["signal"] = raw
        df["size"] = size.astype(float)

        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")
