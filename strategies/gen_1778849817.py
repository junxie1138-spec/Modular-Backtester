from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class VolumeThrustParams:
    mom_window: int = 2
    vol_window: int = 20
    atr_window: int = 14
    trend_window: int = 5
    vol_surge_z: float = 0.5
    trend_thresh: float = 0.6
    base_size: float = 0.45
    size_scale: float = 0.55
    max_size: float = 1.0


class GeneratedStrategy(BaseStrategy[VolumeThrustParams]):
    strategy_id = "gen_1778849817"

    @classmethod
    def params_type(cls):
        return VolumeThrustParams

    def warmup_bars(self, params: VolumeThrustParams) -> int:
        return int(max(params.mom_window, params.vol_window,
                       params.atr_window, params.trend_window)) + 2

    def indicators(self, data: pd.DataFrame, params: VolumeThrustParams) -> pd.DataFrame:
        close = data["close"].astype(float)
        high = data["high"].astype(float)
        low = data["low"].astype(float)
        volume = data["volume"].astype(float)

        mw = max(int(params.mom_window), 1)
        vw = max(int(params.vol_window), 2)
        aw = max(int(params.atr_window), 1)
        tw = max(int(params.trend_window), 1)

        # Short-horizon momentum thrust (the wave-crest displacement)
        mom = close.pct_change(mw)

        # ATR to normalise the thrust into volatility-comparable units
        prev_close = close.shift(1)
        tr = pd.concat(
            [
                (high - low),
                (high - prev_close).abs(),
                (low - prev_close).abs(),
            ],
            axis=1,
        ).max(axis=1)
        atr = tr.rolling(window=aw, min_periods=aw).mean()
        atr_pct = (atr / close).replace([np.inf, -np.inf], np.nan)
        atr_pct = atr_pct.where(atr_pct > 0.0, other=np.nan)
        mom_norm = mom / atr_pct

        # Volume surge z-score: crowd-flow confirmation of the move
        vol_ma = volume.rolling(window=vw, min_periods=vw).mean()
        vol_sd = volume.rolling(window=vw, min_periods=vw).std(ddof=0)
        vol_sd = vol_sd.where(vol_sd > 0.0, other=np.nan)
        vol_z = (volume - vol_ma) / vol_sd

        # Directional consistency over the trend window
        step = np.sign(close.diff())
        dir_consistency = step.rolling(window=tw, min_periods=tw).mean()

        out = pd.DataFrame(
            {
                "mom_norm": mom_norm,
                "vol_z": vol_z,
                "dir_consistency": dir_consistency,
            },
            index=data.index,
        )
        return out

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: VolumeThrustParams,
    ) -> SignalFrame:
        mom_norm = indicators["mom_norm"]
        vol_z = indicators["vol_z"]
        dir_consistency = indicators["dir_consistency"]

        valid = mom_norm.notna() & vol_z.notna() & dir_consistency.notna()
        volume_confirmed = valid & (vol_z > float(params.vol_surge_z))

        long_cond = (
            volume_confirmed
            & (mom_norm > 0.0)
            & (dir_consistency >= float(params.trend_thresh))
        )
        short_cond = (
            volume_confirmed
            & (mom_norm < 0.0)
            & (dir_consistency <= -float(params.trend_thresh))
        )

        signal = pd.Series(0, index=data.index, dtype="int64")
        signal[long_cond] = 1
        signal[short_cond] = -1

        # Signal-scaled position sizing: larger crest x larger surge -> larger size
        surge = (vol_z - float(params.vol_surge_z)).clip(lower=0.0).fillna(0.0)
        thrust = mom_norm.abs().fillna(0.0)
        strength = (surge * thrust).clip(lower=0.0).fillna(0.0)
        strength_unit = 1.0 - np.exp(-strength)  # bounded in [0, 1)

        base = float(params.base_size)
        size = base * (1.0 + float(params.size_scale) * strength_unit)
        size = size.clip(lower=1e-3, upper=float(params.max_size)).fillna(base)

        df = pd.DataFrame(index=data.index)
        df["signal"] = signal.shift(1).fillna(0).astype(int)
        df["size"] = size.shift(1).fillna(base).astype(float)

        return SignalFrame(data=df, signal_column="signal", size_column="size")
