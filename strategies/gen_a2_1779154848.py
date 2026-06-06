from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class GeneratedParams:
    # EWMA span for the leaky-integrator return-tension accumulator (spring stiffness).
    tension_span: int = 10
    # Rolling window over which tension percentile thresholds are measured.
    pct_window: int = 189
    # Upper percentile for entry; lower entry uses (1 - entry_pct).
    entry_pct: float = 0.85
    # Percentile defining the neutral band where the spring is considered relaxed (exit).
    exit_pct: float = 0.50


class GeneratedStrategy(BaseStrategy[GeneratedParams]):
    strategy_id = "gen_a2_1779154848"

    @classmethod
    def params_type(cls) -> type[GeneratedParams]:
        return GeneratedParams

    @staticmethod
    def warmup_bars(params: GeneratedParams) -> int:
        return int(params.pct_window) + int(params.tension_span) + 2

    @staticmethod
    def indicators(data: pd.DataFrame, params: GeneratedParams) -> pd.DataFrame:
        span = max(1, int(params.tension_span))
        win = max(2, int(params.pct_window))
        entry_p = float(params.entry_pct)
        exit_p = float(params.exit_pct)
        # Clamp percentiles into a sane open interval so the bands stay ordered.
        entry_p = min(0.99, max(0.51, entry_p))
        exit_p = min(0.90, max(0.10, exit_p))

        close = data["close"].astype(float)
        ret = close.pct_change().fillna(0.0)

        # Leaky integrator of close-to-close returns: accumulated directional tension.
        tension = ret.ewm(span=span, adjust=False).mean()

        roll = tension.rolling(window=win, min_periods=win)
        upper = roll.quantile(entry_p)
        lower = roll.quantile(1.0 - entry_p)
        mid = roll.quantile(exit_p)

        ind = pd.DataFrame(index=data.index)
        ind["tension"] = tension
        ind["upper"] = upper
        ind["lower"] = lower
        ind["mid"] = mid
        return ind

    @staticmethod
    def generate_signals(
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: GeneratedParams,
    ) -> SignalFrame:
        n = len(data)
        tension = indicators["tension"].to_numpy(dtype=float)
        upper = indicators["upper"].to_numpy(dtype=float)
        lower = indicators["lower"].to_numpy(dtype=float)
        mid = indicators["mid"].to_numpy(dtype=float)

        raw = np.zeros(n, dtype=np.int64)
        state = 0
        for i in range(n):
            u = upper[i]
            lo = lower[i]
            m = mid[i]
            t = tension[i]
            # During warmup the percentile bands are NaN: stay flat, hold no state.
            if not (np.isfinite(u) and np.isfinite(lo) and np.isfinite(m) and np.isfinite(t)):
                state = 0
                raw[i] = 0
                continue

            if state == 0:
                if t >= u:
                    state = 1
                elif t <= lo:
                    state = -1
            elif state == 1:
                # Spring relaxed back to neutral: release the long.
                if t < m:
                    state = 0
                    if t <= lo:
                        state = -1
            else:  # state == -1
                if t > m:
                    state = 0
                    if t >= u:
                        state = 1

            raw[i] = state

        df = pd.DataFrame(index=data.index)
        df["signal"] = raw
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = 1.0
        return SignalFrame(data=df, signal_column="signal", size_column="size")
