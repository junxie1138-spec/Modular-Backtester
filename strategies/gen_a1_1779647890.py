from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class SpringTensionROCAccelParams:
    roc_period: int = 5
    accel_period: int = 2
    tension_lookback: int = 20
    tension_zscore_floor: float = -1.0
    accel_threshold: float = 0.0
    hold_bars: int = 4
    min_size: float = 0.25
    max_size: float = 1.0
    tension_scale: float = 2.0


class GeneratedStrategy(BaseStrategy[SpringTensionROCAccelParams]):
    strategy_id = "gen_a1_1779647890"

    @classmethod
    def params_type(cls):
        return SpringTensionROCAccelParams

    @classmethod
    def warmup_bars(cls, params: SpringTensionROCAccelParams) -> int:
        return int(params.roc_period + params.accel_period + params.tension_lookback + 5)

    def indicators(self, data: pd.DataFrame, params: SpringTensionROCAccelParams) -> pd.DataFrame:
        close = data["close"]

        roc = close.pct_change(params.roc_period)
        roc_accel = roc.diff(params.accel_period)

        ma = close.rolling(params.tension_lookback, min_periods=params.tension_lookback).mean()
        sd = close.rolling(params.tension_lookback, min_periods=params.tension_lookback).std()
        sd_safe = sd.replace(0.0, np.nan)
        tension_z = (close - ma) / sd_safe
        spring_tension = (-tension_z).clip(lower=0.0)

        return pd.DataFrame(
            {
                "roc": roc,
                "roc_accel": roc_accel,
                "roc_accel_prev": roc_accel.shift(1),
                "tension_z": tension_z,
                "spring_tension": spring_tension,
            },
            index=data.index,
        )

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: SpringTensionROCAccelParams,
    ) -> SignalFrame:
        n = len(data)
        signal = np.zeros(n, dtype=np.int64)
        size = np.zeros(n, dtype=np.float64)

        accel = indicators["roc_accel"].to_numpy()
        accel_prev = indicators["roc_accel_prev"].to_numpy()
        tension_z = indicators["tension_z"].to_numpy()
        spring_tension = indicators["spring_tension"].to_numpy()

        scale_div = max(float(params.tension_scale), 1e-6)
        floor = float(params.tension_zscore_floor)
        thr = float(params.accel_threshold)
        hold = max(int(params.hold_bars), 1)
        min_sz = float(params.min_size)
        max_sz = float(params.max_size)
        if max_sz < min_sz:
            max_sz = min_sz

        in_position = False
        bars_held = 0
        position_size = 0.0

        for i in range(n):
            if in_position:
                signal[i] = 1
                size[i] = position_size
                bars_held += 1
                if bars_held >= hold:
                    in_position = False
                    bars_held = 0
                    position_size = 0.0
                continue

            ap = accel_prev[i]
            ac = accel[i]
            tz = tension_z[i]
            st = spring_tension[i]
            if np.isnan(ap) or np.isnan(ac) or np.isnan(tz) or np.isnan(st):
                continue

            tension_ok = tz <= floor
            accel_cross = (ap <= thr) and (ac > thr)
            if tension_ok and accel_cross:
                raw = st / scale_div
                sized = float(np.clip(raw, min_sz, max_sz))
                signal[i] = 1
                size[i] = sized
                in_position = True
                bars_held = 1
                position_size = sized

        df = pd.DataFrame(
            {
                "signal": signal,
                "size": size,
            },
            index=data.index,
        )

        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = df["size"].shift(1).fillna(0.0)
        df.loc[df["size"] <= 0.0, "size"] = 1.0
        return SignalFrame(data=df, signal_column="signal", size_column="size")
