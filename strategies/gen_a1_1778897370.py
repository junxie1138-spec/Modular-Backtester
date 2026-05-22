from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class GenA1Params:
    roc_len: int = 10
    accel_len: int = 5
    noise_len: int = 20
    snr_threshold: float = 1.0
    ma_len: int = 200
    vol_len: int = 20
    target_vol: float = 0.15
    hold_bars: int = 18
    size_min: float = 0.25
    size_max: float = 1.5


class GeneratedStrategy(BaseStrategy[GenA1Params]):
    strategy_id = "gen_a1_1778897370"

    @classmethod
    def params_type(cls) -> type[GenA1Params]:
        return GenA1Params

    def warmup_bars(self, params: GenA1Params) -> int:
        accel_path = params.roc_len + params.accel_len + params.noise_len
        return int(max(params.ma_len, accel_path, params.vol_len)) + 2

    def indicators(self, data: pd.DataFrame, params: GenA1Params) -> pd.DataFrame:
        close = data["close"]
        ind = pd.DataFrame(index=data.index)

        # Rate-of-change and its acceleration (change in ROC over accel_len bars).
        roc = close.pct_change(params.roc_len)
        accel = roc - roc.shift(params.accel_len)

        # Signal-to-noise: normalize acceleration by its own rolling dispersion.
        accel_noise = accel.rolling(params.noise_len, min_periods=params.noise_len).std()
        accel_noise = accel_noise.where(accel_noise > 0.0)
        snr = accel / accel_noise

        ind["roc"] = roc
        ind["accel"] = accel
        ind["snr"] = snr
        ind["ma"] = close.rolling(params.ma_len, min_periods=params.ma_len).mean()

        # Annualized realized volatility for vol-targeted position sizing.
        daily_ret = close.pct_change()
        rv = daily_ret.rolling(params.vol_len, min_periods=params.vol_len).std()
        ind["vol"] = rv * np.sqrt(252.0)
        return ind

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: GenA1Params,
    ) -> SignalFrame:
        n = len(data)
        snr = indicators["snr"].to_numpy(dtype=float)
        roc = indicators["roc"].to_numpy(dtype=float)
        ma = indicators["ma"].to_numpy(dtype=float)
        vol = indicators["vol"].to_numpy(dtype=float)
        close = data["close"].to_numpy(dtype=float)

        # Entry: convex upward momentum (ROC positive AND acceleration SNR high)
        # while price is above the 200-day MA regime gate. NaN-safe.
        entry_ok = np.zeros(n, dtype=bool)
        valid = ~(np.isnan(snr) | np.isnan(roc) | np.isnan(ma) | np.isnan(vol))
        entry_ok[valid] = (
            (snr[valid] > params.snr_threshold)
            & (roc[valid] > 0.0)
            & (close[valid] > ma[valid])
        )

        # Fixed-bar exit: hold exactly hold_bars bars after entry, then flatten.
        signal = np.zeros(n, dtype=int)
        in_pos = False
        entry_idx = 0
        for i in range(n):
            if in_pos:
                if i - entry_idx >= params.hold_bars:
                    in_pos = False
                else:
                    signal[i] = 1
            if not in_pos and entry_ok[i]:
                in_pos = True
                entry_idx = i
                signal[i] = 1

        # Volatility targeting: size inversely with realized vol, clipped.
        with np.errstate(divide="ignore", invalid="ignore"):
            size = params.target_vol / vol
        size = np.clip(size, params.size_min, params.size_max)
        size[~np.isfinite(size)] = params.size_min

        df = pd.DataFrame(index=data.index)
        # MANDATORY one-bar shift: decision on bar N close, fill on bar N+1.
        df["signal"] = (
            pd.Series(signal, index=data.index).shift(1).fillna(0).astype(int)
        )
        df["size"] = (
            pd.Series(size, index=data.index).shift(1).fillna(params.size_min)
        )
        df["size"] = df["size"].clip(lower=params.size_min)
        return SignalFrame(data=df, signal_column="signal", size_column="size")
