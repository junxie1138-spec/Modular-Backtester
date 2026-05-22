from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class GeneratedParams:
    window: int = 10
    epidemic_threshold: float = 0.62
    velocity_lookback: int = 3
    size_gain: float = 2.0
    max_size: float = 1.6


class GeneratedStrategy(BaseStrategy[GeneratedParams]):
    """SI-epidemic trend-infection: trade the direction of a high-and-rising
    return-sign agreement fraction, two-bar confirmed, flip-only exit."""

    strategy_id = "gen_a2_1779155428"

    @classmethod
    def params_type(cls) -> type[GeneratedParams]:
        return GeneratedParams

    @staticmethod
    def warmup_bars(params: GeneratedParams) -> int:
        return int(params.window) + int(params.velocity_lookback) + 5

    @staticmethod
    def indicators(data: pd.DataFrame, params: GeneratedParams) -> pd.DataFrame:
        w = max(2, int(params.window))
        vlb = max(1, int(params.velocity_lookback))

        close = data["close"]
        ret = close.pct_change()
        # net directional move over the window -> prevailing trend direction
        net = close.pct_change(w)
        trend_dir = np.sign(net)

        # fraction of up bars inside the window
        up_frac = (ret > 0).astype(float).rolling(w).mean()
        # "infected fraction": share of window bars whose return sign agrees
        # with the window's net direction (susceptible -> infected by the trend)
        infected = pd.Series(
            np.where(trend_dir > 0, up_frac, 1.0 - up_frac),
            index=close.index,
        )
        # undefined when the window has no net direction
        infected = infected.where(trend_dir != 0)

        # transmission velocity: is the epidemic still spreading?
        velocity = infected - infected.shift(vlb)

        out = pd.DataFrame(index=data.index)
        out["trend_dir"] = trend_dir
        out["infected"] = infected
        out["velocity"] = velocity
        return out

    @staticmethod
    def generate_signals(
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: GeneratedParams,
    ) -> SignalFrame:
        theta = float(params.epidemic_threshold)
        gain = float(params.size_gain)
        max_size = max(1.0, float(params.max_size))

        trend_dir = indicators["trend_dir"].fillna(0.0).to_numpy()
        infected = indicators["infected"].fillna(0.0).to_numpy()
        velocity = indicators["velocity"].fillna(0.0).to_numpy()

        # epidemic established (above threshold) AND still spreading
        spreading = (infected > theta) & (velocity > 0.0)
        long_now = spreading & (trend_dir > 0)
        short_now = spreading & (trend_dir < 0)

        # two-bar confirmation before entry
        n = len(data)
        if n > 0:
            long_prev = np.concatenate(([False], long_now[:-1]))
            short_prev = np.concatenate(([False], short_now[:-1]))
        else:
            long_prev = long_now
            short_prev = short_now
        long_entry = long_now & long_prev
        short_entry = short_now & short_prev

        # flip-only state machine: exit only when the opposite entry fires
        signal = np.zeros(n, dtype=int)
        pos = 0
        for i in range(n):
            if long_entry[i]:
                pos = 1
            elif short_entry[i]:
                pos = -1
            signal[i] = pos

        # viral-load scaled size: stronger infection -> larger position
        load = np.clip(infected - theta, 0.0, None)
        size = np.clip(1.0 + gain * load, 1.0, max_size)

        df = pd.DataFrame(index=data.index)
        df["signal"] = signal
        df["size"] = size.astype(float)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")
