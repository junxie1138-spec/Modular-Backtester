from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class GeneratedParams:
    window: int = 60
    hold_bars: int = 17


class GeneratedStrategy(BaseStrategy[GeneratedParams]):
    strategy_id = "gen_a1_1778904062"

    @classmethod
    def params_type(cls) -> type[GeneratedParams]:
        return GeneratedParams

    @staticmethod
    def warmup_bars(params: GeneratedParams) -> int:
        return int(max(params.window, 2)) * 2 + 1

    @staticmethod
    def indicators(data: pd.DataFrame, params: GeneratedParams) -> pd.DataFrame:
        w = int(max(params.window, 2))
        close = data["close"].astype(float)
        open_ = data["open"].astype(float)

        # Drawdown depth: <= 0, distance below the rolling peak.
        roll_max = close.rolling(w, min_periods=1).max()
        dd = close / roll_max - 1.0

        # Normalize drawdown depth within its own recent cycle: the
        # predator-prey state variable. 0 = deepest drawdown of the window
        # (prey abundant), 1 = shallowest / near peak (prey scarce).
        dd_min = dd.rolling(w, min_periods=2).min()
        dd_max = dd.rolling(w, min_periods=2).max()
        span = (dd_max - dd_min).replace(0.0, np.nan)
        dd_norm = ((dd - dd_min) / span).clip(lower=0.0, upper=1.0).fillna(0.5)

        # Gap measured against recent gap volatility - significance with no
        # extra tunable param (fixed 1.0 std multiplier).
        prev_close = close.shift(1)
        gap = (open_ - prev_close) / prev_close.replace(0.0, np.nan)
        gap_std = gap.rolling(w, min_periods=5).std()
        down_gap = ((gap < -gap_std) & gap_std.notna() & gap.notna())
        up_gap = ((gap > gap_std) & gap_std.notna() & gap.notna())

        out = pd.DataFrame(index=data.index)
        out["dd_norm"] = dd_norm
        out["down_gap"] = down_gap.astype(float).fillna(0.0)
        out["up_gap"] = up_gap.astype(float).fillna(0.0)
        return out

    @staticmethod
    def generate_signals(
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: GeneratedParams,
    ) -> SignalFrame:
        n = len(data)
        dd_norm = indicators["dd_norm"].to_numpy(dtype=float)
        down_gap = indicators["down_gap"].to_numpy(dtype=float)
        up_gap = indicators["up_gap"].to_numpy(dtype=float)

        # Raw entry events: a significant gap at a drawdown-cycle extreme.
        raw = np.zeros(n, dtype=int)
        long_mask = (down_gap > 0.5) & (dd_norm < 0.2)
        short_mask = (up_gap > 0.5) & (dd_norm > 0.8)
        raw[long_mask] = 1
        raw[short_mask] = -1

        # Fixed-bar exit: hold exactly hold_bars bars, then flat. Entry
        # events during an open position are ignored.
        hold = int(max(params.hold_bars, 1))
        signal = np.zeros(n, dtype=int)
        i = 0
        while i < n:
            if raw[i] != 0:
                end = min(i + hold, n)
                signal[i:end] = raw[i]
                i = end
            else:
                i += 1

        df = pd.DataFrame(index=data.index)
        df["signal"] = signal
        df["size"] = 1.0
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")
