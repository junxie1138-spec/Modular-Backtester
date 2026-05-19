from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


# Fixed (non-tunable) hysteresis constants. Keeping these hard-coded enforces
# the <=2 tunable-param twist while still expressing a genuine Schmitt trigger.
_FIRE_THRESHOLD = 2.0   # z-score level that triggers a breakout entry
_ARM_THRESHOLD = 0.5    # z-score band the series must re-enter to re-arm


@dataclass(slots=True)
class GenA2Params:
    lookback: int = 40      # window for the rolling return mean/std
    hold_bars: int = 8      # fixed-bar exit horizon (~1.5 weeks)


class GeneratedStrategy(BaseStrategy[GenA2Params]):
    strategy_id = "gen_a2_1779151305"

    @classmethod
    def params_type(cls) -> type[GenA2Params]:
        return GenA2Params

    @staticmethod
    def warmup_bars(params: GenA2Params) -> int:
        # pct_change consumes one bar, then a rolling window of `lookback`.
        return int(params.lookback) + 2

    @staticmethod
    def indicators(data: pd.DataFrame, params: GenA2Params) -> pd.DataFrame:
        lookback = max(2, int(params.lookback))
        close = data["close"].astype(float)

        ret = close.pct_change()
        mean = ret.rolling(lookback).mean()
        std = ret.rolling(lookback).std()

        # Standardised single-bar return. Guard against a zero/NaN std so the
        # ratio never produces +/-inf.
        safe_std = std.replace(0.0, np.nan)
        z = (ret - mean) / safe_std
        z = z.replace([np.inf, -np.inf], np.nan)

        out = pd.DataFrame(index=data.index)
        out["z"] = z
        return out

    @staticmethod
    def generate_signals(
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: GenA2Params,
    ) -> SignalFrame:
        hold_bars = max(1, int(params.hold_bars))
        z = indicators["z"].to_numpy(dtype=float)
        n = len(z)
        sig = np.zeros(n, dtype=int)

        pos = 0          # current held direction (0 = flat)
        held = 0         # bars elapsed since entry
        armed = True     # hysteresis gate: True once z has been calm

        for i in range(n):
            # --- manage an open position: pure fixed-bar exit ---
            if pos != 0:
                held += 1
                if held >= hold_bars:
                    # fixed-bar exit fires: this bar is flat, no re-entry here
                    pos = 0
                    held = 0
                    continue
                sig[i] = pos
                continue

            # --- flat: update the Schmitt-trigger arm state ---
            zi = z[i]
            if np.isnan(zi):
                continue
            if abs(zi) < _ARM_THRESHOLD:
                armed = True

            # --- fire a breakout entry only while armed ---
            if armed:
                if zi > _FIRE_THRESHOLD:
                    pos = 1
                    held = 0
                    armed = False
                    sig[i] = 1
                elif zi < -_FIRE_THRESHOLD:
                    pos = -1
                    held = 0
                    armed = False
                    sig[i] = -1

        df = pd.DataFrame(index=data.index)
        df["signal"] = sig
        df["size"] = 1.0

        # MANDATORY one-bar shift: decide on bar N close, fill on bar N+1.
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)

        return SignalFrame(data=df, signal_column="signal", size_column="size")
