from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class GeneratedParams:
    capacity: int = 5
    exit_bars: int = 7
    trend_ma: int = 10
    vol_window: int = 8
    target_vol: float = 0.012
    size_floor: float = 0.5
    size_cap: float = 1.5
    ret_eps: float = 0.0


class GeneratedStrategy(BaseStrategy[GeneratedParams]):
    strategy_id = "gen_a1_1778903559"

    @classmethod
    def params_type(cls) -> type[GeneratedParams]:
        return GeneratedParams

    @staticmethod
    def warmup_bars(params: GeneratedParams) -> int:
        return int(max(int(params.trend_ma), int(params.vol_window) + 1))

    @staticmethod
    def indicators(data: pd.DataFrame, params: GeneratedParams) -> pd.DataFrame:
        close = data["close"].astype(float)
        n = len(close)

        ret = close.pct_change()
        eps = float(params.ret_eps)
        sign = pd.Series(0, index=close.index, dtype=int)
        sign[ret > eps] = 1
        sign[ret < -eps] = -1

        cap = int(max(1, int(params.capacity)))
        sign_arr = sign.to_numpy()
        acc = np.zeros(n, dtype=float)
        a = 0
        for i in range(n):
            a += int(sign_arr[i])
            if a > cap:
                a = cap
            elif a < 0:
                a = 0
            acc[i] = float(a)

        sma = close.rolling(int(max(1, int(params.trend_ma)))).mean()
        vol = ret.rolling(int(max(2, int(params.vol_window)))).std()

        out = pd.DataFrame(index=data.index)
        out["ret"] = ret
        out["acc"] = acc
        out["sma"] = sma
        out["vol"] = vol
        return out

    @staticmethod
    def generate_signals(
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: GeneratedParams,
    ) -> SignalFrame:
        close = data["close"].astype(float)
        n = len(close)
        cap = int(max(1, int(params.capacity)))

        acc = indicators["acc"]
        prev_acc = acc.shift(1)
        sma = indicators["sma"]
        vol = indicators["vol"]

        saturated = (acc >= cap) & (prev_acc < cap)
        trend_ok = close > sma
        entry = (saturated & trend_ok).fillna(False).to_numpy()

        exit_bars = int(max(1, int(params.exit_bars)))
        position = np.zeros(n, dtype=int)
        holding = False
        bars_in = 0
        for t in range(n):
            if holding:
                position[t] = 1
                bars_in += 1
                if bars_in >= exit_bars:
                    holding = False
                    bars_in = 0
            elif entry[t]:
                holding = True
                position[t] = 1
                bars_in = 1

        tv = float(params.target_vol)
        scale = tv / vol.replace(0.0, np.nan)
        scale = scale.clip(
            lower=float(params.size_floor), upper=float(params.size_cap)
        )
        scale = scale.fillna(1.0)
        scale = scale.where(scale > 0.0, 1.0).astype(float)

        df = pd.DataFrame(index=data.index)
        df["signal"] = position
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = scale.to_numpy()
        return SignalFrame(data=df, signal_column="signal", size_column="size")
