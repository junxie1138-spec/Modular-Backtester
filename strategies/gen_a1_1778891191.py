from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy

# --- fixed epidemic constants (structural, not tunable) ---
_Z_INFECT = 1.0   # a bar is "infectious" when close sits >1 std above its MA
_BETA = 0.12      # transmission rate: new infections per infectious bar
_GAMMA = 0.05     # recovery rate: infected-fraction decay per bar
_I_IGNITE = 0.25  # infected-fraction threshold marking epidemic ignition


@dataclass(slots=True)
class GeneratedParams:
    ma_window: int = 20
    profit_target: float = 0.05


class GeneratedStrategy(BaseStrategy[GeneratedParams]):
    strategy_id = "gen_a1_1778891191"

    @classmethod
    def params_type(cls):
        return GeneratedParams

    @staticmethod
    def warmup_bars(params: GeneratedParams) -> int:
        return int(params.ma_window) + 2

    @staticmethod
    def indicators(data: pd.DataFrame, params: GeneratedParams) -> pd.DataFrame:
        w = max(2, int(params.ma_window))
        close = data["close"].astype(float)

        ma = close.rolling(w).mean()
        sd = close.rolling(w).std(ddof=0)
        z = (close - ma) / sd.replace(0.0, np.nan)
        z = z.fillna(0.0)

        # infectious input: a price breakout more than Z_INFECT std above the MA
        infectious = (z > _Z_INFECT).to_numpy(dtype=float)

        # forced susceptible-infected dynamics:
        #   I_t = I_{t-1} + BETA*(1 - I_{t-1})*x_t - GAMMA*I_{t-1}
        # each infectious bar recruits new infections from the susceptible pool
        # (1 - I); the pool depletes as I rises and recovery continuously decays I.
        n = len(close)
        infected = np.zeros(n, dtype=float)
        prev = 0.0
        for i in range(n):
            x = infectious[i]
            cur = prev + _BETA * (1.0 - prev) * x - _GAMMA * prev
            if cur < 0.0:
                cur = 0.0
            elif cur > 1.0:
                cur = 1.0
            infected[i] = cur
            prev = cur

        out = pd.DataFrame(index=data.index)
        out["z"] = z
        out["infected"] = infected
        out["susceptible"] = 1.0 - infected
        out["infectious"] = infectious
        return out

    @staticmethod
    def generate_signals(data: pd.DataFrame, indicators: pd.DataFrame,
                         ctx: StrategyContext, params: GeneratedParams) -> SignalFrame:
        w = max(2, int(params.ma_window))
        target = float(params.profit_target)
        time_stop = max(3, w // 2)  # ~1-2 week hold for typical ma_window

        z = indicators["z"].to_numpy(dtype=float)
        infected = indicators["infected"].to_numpy(dtype=float)
        close = data["close"].to_numpy(dtype=float)
        n = len(close)

        prev_infected = np.zeros(n, dtype=float)
        if n > 1:
            prev_infected[1:] = infected[:-1]

        # epidemic ignition: the infected fraction crosses up through I_IGNITE
        # while price is genuinely above its MA (z > 0) -> breakout confirmed,
        # susceptible pool still large (S = 1 - I ~= 0.73 at the crossing).
        entry_raw = (
            (infected > _I_IGNITE)
            & (prev_infected <= _I_IGNITE)
            & (z > 0.0)
        )

        signal = np.zeros(n, dtype=int)
        position = 0
        entry_price = 0.0
        bars_held = 0

        for i in range(n):
            if position == 0:
                if entry_raw[i]:
                    position = 1
                    entry_price = close[i]
                    bars_held = 0
                    signal[i] = 1
                else:
                    signal[i] = 0
            else:
                bars_held += 1
                gain = (close[i] / entry_price - 1.0) if entry_price > 0.0 else 0.0
                # exit at +target gain OR after time_stop bars, whichever first
                if gain >= target or bars_held >= time_stop:
                    position = 0
                    entry_price = 0.0
                    bars_held = 0
                    signal[i] = 0
                else:
                    signal[i] = 1

        df = pd.DataFrame(index=data.index)
        df["signal"] = signal
        df["size"] = 1.0
        # mandatory one-bar shift: decide on bar N close, fill on bar N+1
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")
