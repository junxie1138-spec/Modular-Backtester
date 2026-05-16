from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class SpringRegimeParams:
    window: int = 20
    stiffness_threshold: float = 0.3


class GeneratedStrategy(BaseStrategy[SpringRegimeParams]):
    """Regime-switching long-only strategy driven by a spring-stiffness estimate.

    Displacement x = close - SMA(close, window) is treated as the extension of a
    spring from its equilibrium. Velocity is x.diff(); acceleration is the ROC of
    that velocity (x.diff().diff()) and approximates the restoring force.

    A real spring obeys force = -k * x, so acceleration and -x are positively
    correlated. The rolling correlation of acceleration with -x is therefore a
    direct proxy for spring stiffness k:

      * stiffness > threshold  -> SPRING regime. A restoring force is active, so
        a downward-stretched displacement (x < 0) is expected to be pulled back
        up. Raw directional desire = +1 when x < 0, -1 when x > 0.
      * stiffness <= threshold -> INERTIAL regime. No restoring force; momentum
        carries. Raw directional desire = sign of the window-length ROC.

    Long-only with a signal-reversal exit: enter long when the raw desire turns
    +1, and exit only when it flips to -1 (the entry condition reversing).
    """

    strategy_id = "gen_a1_1778908072"

    @classmethod
    def params_type(cls) -> type[SpringRegimeParams]:
        return SpringRegimeParams

    def warmup_bars(self, params: SpringRegimeParams) -> int:
        # SMA needs `window`; x.diff().diff() adds 2; the rolling corr over the
        # acceleration series adds another `window`. 2*window + 2 is the true
        # longest lookback; add slack.
        return 2 * int(params.window) + 5

    def indicators(self, data: pd.DataFrame, params: SpringRegimeParams) -> pd.DataFrame:
        w = int(params.window)
        if w < 2:
            w = 2
        close = data["close"]

        sma = close.rolling(w).mean()
        x = close - sma                       # displacement from equilibrium
        vel = x.diff()                        # velocity
        accel = vel.diff()                    # acceleration ~ restoring force
        stiffness = accel.rolling(w).corr(-x)  # spring constant proxy in [-1, 1]
        roc = close / close.shift(w) - 1.0     # trend velocity for inertial leg

        out = pd.DataFrame(index=data.index)
        out["x"] = x
        out["stiffness"] = stiffness
        out["roc"] = roc
        return out

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: SpringRegimeParams,
    ) -> SignalFrame:
        n = len(data)
        thr = float(params.stiffness_threshold)

        xv = indicators["x"].to_numpy(dtype=float)
        # NaN stiffness (warmup or zero-variance window) -> 0.0 -> inertial regime.
        stiffv = indicators["stiffness"].fillna(0.0).to_numpy(dtype=float)
        rocv = indicators["roc"].to_numpy(dtype=float)

        pos = np.zeros(n, dtype=int)
        state = 0  # 0 = flat, 1 = long

        for i in range(n):
            desire = 0
            xi = xv[i]
            if not np.isnan(xi):
                if stiffv[i] > thr:
                    # Spring regime: trade the elastic restoring force.
                    if xi < 0.0:
                        desire = 1
                    elif xi > 0.0:
                        desire = -1
                else:
                    # Inertial regime: trade momentum.
                    ri = rocv[i]
                    if not np.isnan(ri):
                        if ri > 0.0:
                            desire = 1
                        elif ri < 0.0:
                            desire = -1

            # Signal-reversal exit: enter long on +1, hold through 0 and +1,
            # exit only when the entry condition flips to -1.
            if state == 0:
                if desire == 1:
                    state = 1
            else:
                if desire == -1:
                    state = 0
            pos[i] = state

        df = pd.DataFrame(index=data.index)
        df["signal"] = pos
        # Mandatory one-bar shift: decide on bar N close, fill on bar N+1.
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = 1.0
        return SignalFrame(data=df, signal_column="signal", size_column="size")
