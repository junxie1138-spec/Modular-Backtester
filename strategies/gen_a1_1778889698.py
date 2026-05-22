from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class GeneratedParams:
    """Tunable parameters (twist: <=2 tunable params)."""
    window: int = 15
    asym_threshold: float = 0.58


class GeneratedStrategy(BaseStrategy[GeneratedParams]):
    """Laminar-trend long: enter when upside semivariance dominates downside
    semivariance; exit on a rolling-high ATR trailing stop.
    """

    strategy_id = "gen_a1_1778889698"

    # ATR multiple for the trailing stop. Held fixed so the strategy keeps
    # at most two tunable params per the hard twist.
    _ATR_MULT = 1.5

    @classmethod
    def params_type(cls) -> type[GeneratedParams]:
        return GeneratedParams

    @staticmethod
    def warmup_bars(params: GeneratedParams) -> int:
        # pct_change()/shift(1) consume one bar before the rolling window.
        return int(params.window) + 1

    def indicators(self, data: pd.DataFrame, params: GeneratedParams) -> pd.DataFrame:
        window = max(int(params.window), 2)
        close = data["close"]
        high = data["high"]
        low = data["low"]

        # Decompose daily return variance into upside vs downside semivariance.
        ret = close.pct_change()
        up = ret.clip(lower=0.0)
        down = ret.clip(upper=0.0)
        up_var = up.pow(2).rolling(window).mean()
        down_var = down.pow(2).rolling(window).mean()
        # Share of realized variance that is upside; 1e-12 guards flat markets.
        asym = up_var / (up_var + down_var + 1e-12)

        # Average True Range for the trailing stop.
        prev_close = close.shift(1)
        tr = pd.concat(
            [
                (high - low),
                (high - prev_close).abs(),
                (low - prev_close).abs(),
            ],
            axis=1,
        ).max(axis=1)
        atr = tr.rolling(window).mean()

        out = pd.DataFrame(index=data.index)
        out["asym"] = asym
        out["atr"] = atr
        return out

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: GeneratedParams,
    ) -> SignalFrame:
        close = data["close"].to_numpy(dtype=float)
        asym = indicators["asym"].to_numpy(dtype=float)
        atr = indicators["atr"].to_numpy(dtype=float)
        n = len(close)

        threshold = float(params.asym_threshold)
        k = float(self._ATR_MULT)

        signal = np.zeros(n, dtype=np.int64)
        in_pos = False
        hw = 0.0  # highest close since entry (high-water mark)

        # Path-dependent: rolling-high ATR trailing stop needs a bar loop.
        for i in range(n):
            a = asym[i]
            v = atr[i]
            ready = (
                np.isfinite(a)
                and np.isfinite(v)
                and np.isfinite(close[i])
            )
            if not in_pos:
                if ready and a > threshold:
                    in_pos = True
                    hw = close[i]
                    signal[i] = 1
            else:
                if close[i] > hw:
                    hw = close[i]  # stop only ratchets up
                stop = hw - k * v
                if (not ready) or (close[i] < stop):
                    in_pos = False
                    signal[i] = 0
                else:
                    signal[i] = 1

        df = pd.DataFrame(index=data.index)
        df["signal"] = signal
        df["size"] = 1.0
        # MANDATORY one-bar shift: decide on bar N close, fill on bar N+1.
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")
