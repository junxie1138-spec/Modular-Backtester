from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class GenA1Params:
    vol_window: int = 5
    vol_avg_window: int = 60
    vol_enter_ratio: float = 0.85
    vol_exit_ratio: float = 1.10
    thrust_window: int = 3
    volume_ma_window: int = 20
    thrust_enter: float = 0.005
    thrust_exit: float = 0.0
    vol_target_annual: float = 0.15
    min_size: float = 0.10
    max_size: float = 1.00


class GeneratedStrategy(BaseStrategy[GenA1Params]):
    strategy_id = "gen_a1_1779654063"

    @classmethod
    def params_type(cls):
        return GenA1Params

    @staticmethod
    def warmup_bars(params: GenA1Params) -> int:
        return int(max(
            params.vol_window + params.vol_avg_window,
            params.volume_ma_window + params.thrust_window,
        ) + 2)

    def indicators(self, data: pd.DataFrame, params: GenA1Params) -> pd.DataFrame:
        close = data["close"]
        volume = data["volume"]
        ret = close.pct_change()

        rv = ret.rolling(params.vol_window).std() * np.sqrt(252.0)
        rv_avg = rv.rolling(params.vol_avg_window).mean()
        rv_ratio = rv / rv_avg.replace(0.0, np.nan)

        vol_ma = volume.rolling(params.volume_ma_window).mean()
        vol_z = volume / vol_ma.replace(0.0, np.nan)
        weighted_ret = ret * vol_z
        thrust = weighted_ret.rolling(params.thrust_window).sum()

        rv_safe = rv.replace(0.0, np.nan)
        size_series = (params.vol_target_annual / rv_safe).clip(
            lower=params.min_size, upper=params.max_size
        ).fillna(params.min_size)

        out = pd.DataFrame({
            "rv_ratio": rv_ratio,
            "thrust": thrust,
            "size": size_series,
        }, index=data.index)
        return out

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame,
                         ctx: StrategyContext, params: GenA1Params) -> SignalFrame:
        rv_ratio = indicators["rv_ratio"].to_numpy()
        thrust = indicators["thrust"].to_numpy()
        size_arr = indicators["size"].fillna(params.min_size).to_numpy()

        n = len(data)
        sig = np.zeros(n, dtype=np.int64)

        vol_ok = False
        thrust_ok = False

        for i in range(n):
            r = rv_ratio[i]
            t = thrust[i]

            if not (np.isnan(r) or np.isnan(t)):
                if (not vol_ok) and r < params.vol_enter_ratio:
                    vol_ok = True
                elif vol_ok and r > params.vol_exit_ratio:
                    vol_ok = False

                if (not thrust_ok) and t > params.thrust_enter:
                    thrust_ok = True
                elif thrust_ok and t < params.thrust_exit:
                    thrust_ok = False

            sig[i] = 1 if (vol_ok and thrust_ok) else 0

        df = pd.DataFrame({
            "signal": sig,
            "size": size_arr,
        }, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = df["size"].shift(1).fillna(params.min_size)

        return SignalFrame(data=df, signal_column="signal", size_column="size")
