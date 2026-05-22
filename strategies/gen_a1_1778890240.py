from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy

# Hardcoded structural constants (kept off the param surface to honor the
# <=2 tunable-param twist).
_VOL_WINDOW = 20      # realized-vol lookback (daily return std)
_VOL_MA_WINDOW = 100  # equilibrium window for the vol z-score
_ATR_WINDOW = 14      # ATR lookback for the fixed vol-stop
_MAX_HOLD = 10        # time cap (~2 weeks) to keep the holding horizon bounded
_TARGET_DVOL = 0.01   # target daily volatility for inverse-vol sizing


@dataclass(slots=True)
class GeneratedParams:
    vol_z_threshold: float = 1.0  # spring-tension threshold for the vol z-score
    atr_stop_k: float = 2.5       # k in the fixed entry - k*ATR vol-stop


class GeneratedStrategy(BaseStrategy[GeneratedParams]):
    strategy_id = "gen_a1_1778890240"

    @classmethod
    def params_type(cls):
        return GeneratedParams

    @staticmethod
    def warmup_bars(params: GeneratedParams) -> int:
        # pct_change (+1) -> rolling _VOL_WINDOW -> rolling _VOL_MA_WINDOW.
        return _VOL_WINDOW + _VOL_MA_WINDOW + 5

    @staticmethod
    def indicators(data: pd.DataFrame, params: GeneratedParams) -> pd.DataFrame:
        close = data["close"]
        high = data["high"]
        low = data["low"]

        ret = close.pct_change()
        rvol = ret.rolling(_VOL_WINDOW).std()

        # Volatility equilibrium and its dispersion -> spring-displacement z.
        vol_mean = rvol.rolling(_VOL_MA_WINDOW).mean()
        vol_sd = rvol.rolling(_VOL_MA_WINDOW).std()
        vol_z = (rvol - vol_mean) / vol_sd.replace(0.0, np.nan)

        # ATR for the fixed volatility-stop.
        prev_close = close.shift(1)
        tr = pd.concat(
            [
                (high - low),
                (high - prev_close).abs(),
                (low - prev_close).abs(),
            ],
            axis=1,
        ).max(axis=1)
        atr = tr.rolling(_ATR_WINDOW).mean()

        # Inverse-volatility position size (volatility-targeting hallmark).
        inv_vol = _TARGET_DVOL / rvol.replace(0.0, np.nan)
        inv_vol = inv_vol.clip(lower=0.1, upper=1.0)

        out = pd.DataFrame(index=data.index)
        out["rvol"] = rvol
        out["vol_z"] = vol_z
        out["atr"] = atr
        out["size"] = inv_vol.fillna(0.1)
        return out

    @staticmethod
    def generate_signals(
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: GeneratedParams,
    ) -> SignalFrame:
        close = data["close"].to_numpy(dtype=float)
        vol_z = indicators["vol_z"].to_numpy(dtype=float)
        atr = indicators["atr"].to_numpy(dtype=float)
        size_arr = indicators["size"].to_numpy(dtype=float)

        n = len(close)
        raw = np.zeros(n, dtype=int)

        thr = float(params.vol_z_threshold)
        k = float(params.atr_stop_k)

        pos = 0
        stop = 0.0
        bars_held = 0

        for i in range(1, n):
            z = vol_z[i]
            z_prev = vol_z[i - 1]
            a = atr[i]
            c = close[i]

            if pos == 0:
                # Spring releases: vol z-score recoils back down through the
                # tension threshold from above.
                trigger = (
                    np.isfinite(z)
                    and np.isfinite(z_prev)
                    and np.isfinite(a)
                    and np.isfinite(c)
                    and z_prev >= thr
                    and z < thr
                )
                if trigger:
                    pos = 1
                    stop = c - k * a  # fixed (non-trailing) vol-stop
                    bars_held = 0
                    raw[i] = 1
                else:
                    raw[i] = 0
            else:
                bars_held += 1
                if c <= stop or bars_held >= _MAX_HOLD:
                    pos = 0
                    raw[i] = 0
                else:
                    raw[i] = 1

        df = pd.DataFrame(index=data.index)
        df["signal"] = raw
        # MANDATORY one-bar shift: decide on bar N close, fill on bar N+1.
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)

        size = pd.Series(size_arr, index=data.index).fillna(0.1)
        size = size.clip(lower=0.1, upper=1.0)
        df["size"] = size

        return SignalFrame(data=df, signal_column="signal", size_column="size")
