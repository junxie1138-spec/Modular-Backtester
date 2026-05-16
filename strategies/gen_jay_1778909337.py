from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class Params:
    roc_period: int = 5
    roc_oversold: float = -0.015
    accel_smooth: int = 3
    ma_period: int = 200
    size: float = 0.95


class GeneratedStrategy(BaseStrategy[Params]):
    strategy_id = "gen_jay_1778909337"

    @classmethod
    def params_type(cls):
        return Params

    @staticmethod
    def warmup_bars(params: Params) -> int:
        return params.ma_period + params.roc_period + params.accel_smooth + 2

    @staticmethod
    def indicators(data: pd.DataFrame, params: Params) -> pd.DataFrame:
        close = data["close"]
        ma200 = close.rolling(params.ma_period).mean()
        roc = close.pct_change(params.roc_period)
        roc_accel = roc.diff(1)
        roc_accel_smooth = roc_accel.rolling(params.accel_smooth).mean()
        return pd.DataFrame(
            {
                "ma200": ma200,
                "roc": roc,
                "roc_accel_smooth": roc_accel_smooth,
            },
            index=data.index,
        )

    @staticmethod
    def generate_signals(
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: Params,
    ) -> SignalFrame:
        close = data["close"].values
        ma200 = indicators["ma200"].values
        roc = indicators["roc"].values
        accel = indicators["roc_accel_smooth"].values

        n = len(data)
        signal = np.zeros(n, dtype=int)
        in_position = False

        for i in range(n):
            if np.isnan(accel[i]) or np.isnan(ma200[i]) or np.isnan(roc[i]):
                continue

            bull_regime = close[i] > ma200[i]
            oversold = roc[i] < params.roc_oversold
            accel_pos = accel[i] > 0.0
            accel_prev_neg = (
                i > 0
                and not np.isnan(accel[i - 1])
                and accel[i - 1] <= 0.0
            )
            accel_cross_up = accel_pos and accel_prev_neg

            if not in_position:
                if bull_regime and oversold and accel_cross_up:
                    signal[i] = 1
                    in_position = True
            else:
                exit_now = (not bull_regime) or (roc[i] >= 0.0) or (accel[i] < 0.0)
                if exit_now:
                    signal[i] = 0
                    in_position = False
                else:
                    signal[i] = 1

        df = pd.DataFrame({"signal": signal, "size": params.size}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")
