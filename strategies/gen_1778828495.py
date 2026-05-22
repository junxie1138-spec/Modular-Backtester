from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class SpringTensionParams:
    lookback: int = 20
    velocity_window: int = 3
    tension_threshold: float = 1.5
    holding_bars: int = 18


class GeneratedStrategy(BaseStrategy[SpringTensionParams]):
    strategy_id = "gen_1778828495"

    @classmethod
    def params_type(cls):
        return SpringTensionParams

    def warmup_bars(self, params: SpringTensionParams) -> int:
        return int(params.lookback) + int(params.velocity_window) + 2

    def indicators(self, data: pd.DataFrame, params: SpringTensionParams) -> pd.DataFrame:
        close = data["close"].astype(float)
        returns = close.pct_change()

        K = int(params.lookback)
        V = int(params.velocity_window)
        if K < 2:
            K = 2
        if V < 1:
            V = 1

        displacement = returns.rolling(window=K, min_periods=K).sum()
        vol = returns.rolling(window=K, min_periods=K).std(ddof=0)

        denom = vol * np.sqrt(float(K))
        denom = denom.where(denom > 0.0, other=np.nan)

        tension = -displacement / denom
        velocity = returns.rolling(window=V, min_periods=V).mean()

        out = pd.DataFrame(
            {
                "returns": returns,
                "displacement": displacement,
                "vol": vol,
                "tension": tension,
                "velocity": velocity,
            },
            index=data.index,
        )
        return out

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: SpringTensionParams,
    ) -> SignalFrame:
        tension = indicators["tension"].to_numpy(dtype=float, copy=False)
        velocity = indicators["velocity"].to_numpy(dtype=float, copy=False)

        threshold = float(params.tension_threshold)
        hold = int(params.holding_bars)
        if hold < 1:
            hold = 1

        valid = np.isfinite(tension) & np.isfinite(velocity)
        entry_cond = valid & (tension > threshold) & (velocity > 0.0)

        n = int(len(data))
        position = np.zeros(n, dtype=np.int64)

        i = 0
        while i < n:
            if entry_cond[i]:
                end = i + hold
                if end > n:
                    end = n
                position[i:end] = 1
                i = end
            else:
                i += 1

        raw = pd.Series(position, index=data.index, dtype="int64")
        signal = raw.shift(1).fillna(0).astype(int)
        size = pd.Series(1.0, index=data.index, dtype=float)

        df = pd.DataFrame({"signal": signal, "size": size}, index=data.index)
        return SignalFrame(data=df, signal_column="signal", size_column="size")
