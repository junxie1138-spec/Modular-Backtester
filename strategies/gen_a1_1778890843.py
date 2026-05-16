from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class Params:
    ac_window: int = 20
    bias_window: int = 5
    rho_threshold: float = 0.10
    bias_threshold: float = 0.02
    size_floor: float = 0.5


class GeneratedStrategy(BaseStrategy[Params]):
    strategy_id = "gen_a1_1778890843"

    @classmethod
    def params_type(cls) -> type[Params]:
        return Params

    @staticmethod
    def warmup_bars(params: Params) -> int:
        return int(max(params.ac_window + 1, params.bias_window) + 2)

    @staticmethod
    def indicators(data: pd.DataFrame, params: Params) -> pd.DataFrame:
        high = data["high"].astype(float)
        low = data["low"].astype(float)
        close = data["close"].astype(float)

        rng = high - low
        # intrabar close location in [0, 1]; zero/degenerate range -> neutral 0.5
        loc = (close - low) / rng.where(rng > 0.0, np.nan)
        loc = loc.clip(0.0, 1.0).fillna(0.5)
        c = loc - 0.5  # centered close-location, in [-0.5, 0.5]

        w = max(int(params.ac_window), 2)
        m = max(int(params.bias_window), 1)

        c_lag = c.shift(1)
        # rolling lag-1 autocorrelation of the centered close-location series
        rho = c.rolling(w).corr(c_lag)
        rho = rho.replace([np.inf, -np.inf], np.nan).fillna(0.0)

        # recent directional bias of close-location (upper vs lower range)
        cbar = c.rolling(m).mean().fillna(0.0)

        out = pd.DataFrame(index=data.index)
        out["loc"] = loc
        out["c"] = c
        out["rho"] = rho
        out["cbar"] = cbar
        return out

    @staticmethod
    def generate_signals(
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: Params,
    ) -> SignalFrame:
        rho = indicators["rho"].fillna(0.0)
        cbar = indicators["cbar"].fillna(0.0)

        # Symmetric entry/exit rule: a single boolean drives both sides.
        # Enter long while close-location control is coherent (positively
        # autocorrelated) AND biased to the upper range. Signal-reversal
        # exit: drive signal to 0 the instant that exact condition flips.
        entry = (rho > float(params.rho_threshold)) & (
            cbar > float(params.bias_threshold)
        )
        raw = entry.astype(int)

        # spring-tension-scaled conviction: stronger coherence -> larger size,
        # bounded in [size_floor, 1.0]
        floor = float(params.size_floor)
        tension = rho.clip(0.0, 1.0)
        size = floor + (1.0 - floor) * tension
        size = size.clip(lower=floor, upper=1.0)

        df = pd.DataFrame(index=data.index)
        # MANDATORY one-bar shift: decide on bar N close, fill on N+1
        df["signal"] = raw.shift(1).fillna(0).astype(int)
        df["size"] = size.shift(1).fillna(floor).clip(lower=0.01).astype(float)

        return SignalFrame(data=df, signal_column="signal", size_column="size")
