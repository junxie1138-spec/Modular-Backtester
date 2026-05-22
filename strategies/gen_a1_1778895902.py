from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class ElasticRecoilParams:
    ma_period: int = 5
    recoil_window: int = 4
    recoil_ratio: float = 0.6
    entry_z_max: float = -0.5
    profit_target: float = 0.03
    time_stop: int = 5
    base_size: float = 1.0


class GeneratedStrategy(BaseStrategy[ElasticRecoilParams]):
    """Drawdown-recovery long: enter when the distance-from-MA z-score rebounds
    at a velocity comparable to the peak descent velocity that created the dip
    (elastic recoil). Exit at a profit target or a fixed time-stop.
    """

    strategy_id = "gen_a1_1778895902"

    @classmethod
    def params_type(cls) -> type[ElasticRecoilParams]:
        return ElasticRecoilParams

    @staticmethod
    def warmup_bars(params: ElasticRecoilParams) -> int:
        # ma/std window + one bar consumed by .diff() + descent rolling window.
        return int(params.ma_period) + int(params.recoil_window) + 1

    def indicators(self, data: pd.DataFrame, params: ElasticRecoilParams) -> pd.DataFrame:
        close = data["close"]

        ma = close.rolling(params.ma_period, min_periods=params.ma_period).mean()
        sd = close.rolling(params.ma_period, min_periods=params.ma_period).std(ddof=0)
        z = (close - ma) / sd.replace(0.0, np.nan)

        zdiff = z.diff()
        # descent: largest single-bar drop in z over the recent window (positive).
        descent = (-zdiff).rolling(
            params.recoil_window, min_periods=params.recoil_window
        ).max()
        # recoil ratio: current upward velocity vs peak descent velocity.
        ratio = zdiff / descent.replace(0.0, np.nan)

        cond = (
            (z < params.entry_z_max)
            & (zdiff > 0.0)
            & (descent > 0.0)
            & (ratio >= params.recoil_ratio)
        )
        entry_flag = cond.fillna(False)

        denom = params.recoil_ratio if params.recoil_ratio != 0.0 else 1.0
        strength = (ratio / denom).clip(lower=1.0, upper=2.5)
        strength = strength.where(entry_flag, 1.0).fillna(1.0)

        out = pd.DataFrame(index=data.index)
        out["z"] = z
        out["zdiff"] = zdiff
        out["descent"] = descent
        out["ratio"] = ratio
        out["entry_flag"] = entry_flag.astype(bool)
        out["size_strength"] = strength
        return out

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: ElasticRecoilParams,
    ) -> SignalFrame:
        close = data["close"].to_numpy(dtype=float)
        entry_flag = indicators["entry_flag"].to_numpy()
        strength = indicators["size_strength"].to_numpy(dtype=float)
        n = len(close)

        signal = np.zeros(n, dtype=int)
        size = np.full(n, float(params.base_size), dtype=float)

        in_pos = False
        entry_idx = -1
        entry_price = 0.0

        for i in range(n):
            if in_pos:
                bars_held = i - entry_idx
                gain = (close[i] / entry_price - 1.0) if entry_price > 0.0 else 0.0
                if gain >= params.profit_target or bars_held >= params.time_stop:
                    in_pos = False
                    signal[i] = 0
                else:
                    signal[i] = 1
            else:
                if bool(entry_flag[i]) and np.isfinite(close[i]) and close[i] > 0.0:
                    in_pos = True
                    entry_idx = i
                    entry_price = close[i]
                    signal[i] = 1
                    s = strength[i]
                    if not np.isfinite(s) or s <= 0.0:
                        s = 1.0
                    size[i] = float(params.base_size) * float(s)

        df = pd.DataFrame(index=data.index)
        df["signal"] = (
            pd.Series(signal, index=data.index).shift(1).fillna(0).astype(int)
        )
        df["size"] = (
            pd.Series(size, index=data.index)
            .shift(1)
            .fillna(float(params.base_size))
        )
        return SignalFrame(data=df, signal_column="signal", size_column="size")
