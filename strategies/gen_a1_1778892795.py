from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class ShockwaveGapParams:
    roc_period: int = 3
    accel_lag: int = 2
    gap_std_window: int = 20
    gap_z_thresh: float = 0.8
    gap_lookback: int = 3
    accel_thresh: float = 0.0
    ma_period: int = 200
    base_size: float = 1.0
    size_gain: float = 0.4
    max_size_mult: float = 2.0


class GeneratedStrategy(BaseStrategy[ShockwaveGapParams]):
    """Gap-disturbance primed ROC-acceleration shockwave entry, 200-MA gated."""

    strategy_id = "gen_a1_1778892795"

    @classmethod
    def params_type(cls) -> type[ShockwaveGapParams]:
        return ShockwaveGapParams

    def warmup_bars(self, params: ShockwaveGapParams) -> int:
        return int(max(
            params.ma_period,
            params.gap_std_window + 1,
            params.roc_period + params.accel_lag + 1,
            params.gap_lookback + 1,
        )) + 1

    def indicators(self, data: pd.DataFrame, params: ShockwaveGapParams) -> pd.DataFrame:
        close = data["close"]
        open_ = data["open"]
        prev_close = close.shift(1)

        # Overnight gap as the traffic disturbance.
        gap = (open_ - prev_close) / prev_close.replace(0.0, np.nan)
        gap_std = gap.rolling(params.gap_std_window, min_periods=params.gap_std_window).std()
        gap_z = gap / gap_std.replace(0.0, np.nan)

        # Rate-of-change acceleration of the close price (the release shockwave).
        roc = close.pct_change(params.roc_period)
        accel = roc - roc.shift(params.accel_lag)
        accel_std = accel.rolling(params.gap_std_window, min_periods=params.gap_std_window).std()
        accel_z = accel / accel_std.replace(0.0, np.nan)

        # 200-day MA regime filter (hard twist).
        ma = close.rolling(params.ma_period, min_periods=params.ma_period).mean()
        regime = (close > ma).astype(float)

        # A fresh down-gap disturbance primes a short-lived entry window.
        disturbance = (gap_z < -abs(params.gap_z_thresh)).astype(float)
        primed = disturbance.rolling(params.gap_lookback, min_periods=1).max()

        out = pd.DataFrame(index=data.index)
        out["gap_z"] = gap_z
        out["roc"] = roc
        out["accel"] = accel
        out["accel_z"] = accel_z
        out["ma"] = ma
        out["regime"] = regime
        out["primed"] = primed.fillna(0.0)
        return out

    def generate_signals(self, data, indicators, ctx, params) -> SignalFrame:
        idx = data.index
        n = len(idx)

        accel = indicators["accel"].to_numpy(dtype=float)
        accel_z = indicators["accel_z"].to_numpy(dtype=float)
        ma = indicators["ma"].to_numpy(dtype=float)
        regime = indicators["regime"].to_numpy(dtype=float)
        primed = indicators["primed"].to_numpy(dtype=float)

        thresh = float(params.accel_thresh)
        # NaN accel compares False, so release is NaN-safe.
        release = np.greater(accel, thresh)

        rising = np.zeros(n, dtype=bool)
        if n > 1:
            rising[1:] = release[1:] & ~release[:-1]

        valid = (~np.isnan(accel)) & (~np.isnan(ma))

        # Signal-reversal exit: hold long while acceleration stays positive,
        # flatten the moment that entry condition flips off.
        signal = np.zeros(n, dtype=np.int64)
        position = 0
        for i in range(n):
            if not valid[i]:
                position = 0
                signal[i] = 0
                continue
            if position == 0:
                if regime[i] > 0.5 and primed[i] > 0.5 and rising[i]:
                    position = 1
            else:
                if not release[i]:
                    position = 0
            signal[i] = position

        # Size scaled by shockwave strength, always a positive finite float.
        strength = np.nan_to_num(accel_z, nan=0.0, posinf=0.0, neginf=0.0)
        size = params.base_size * np.clip(
            1.0 + params.size_gain * np.abs(strength),
            0.5,
            float(params.max_size_mult),
        )
        size = np.where(np.isfinite(size) & (size > 0.0), size, params.base_size)

        df = pd.DataFrame(index=idx)
        df["signal"] = pd.Series(signal, index=idx).shift(1).fillna(0).astype(int)
        df["size"] = size.astype(float)
        return SignalFrame(data=df, signal_column="signal", size_column="size")
