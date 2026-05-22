from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class GapAutocorrParams:
    gap_window: int = 60
    ma_window: int = 200
    hold_bars: int = 8
    gap_threshold: float = 0.0015
    ac_arm: float = 0.10


class GeneratedStrategy(BaseStrategy[GapAutocorrParams]):
    strategy_id = "gen_a2_1779153580"

    @classmethod
    def params_type(cls) -> type[GapAutocorrParams]:
        return GapAutocorrParams

    @staticmethod
    def warmup_bars(params: GapAutocorrParams) -> int:
        return int(max(params.ma_window, params.gap_window + 2)) + 1

    @staticmethod
    def indicators(data: pd.DataFrame, params: GapAutocorrParams) -> pd.DataFrame:
        close = data["close"].astype(float)
        open_ = data["open"].astype(float)

        prev_close = close.shift(1)
        gap = open_ / prev_close - 1.0
        gap = gap.replace([np.inf, -np.inf], np.nan)
        gap_lag = gap.shift(1)

        w = max(int(params.gap_window), 5)
        ac = gap.rolling(window=w, min_periods=w).corr(gap_lag)
        ac = ac.replace([np.inf, -np.inf], np.nan)

        ma = close.rolling(window=max(int(params.ma_window), 2),
                           min_periods=max(int(params.ma_window), 2)).mean()

        out = pd.DataFrame(index=data.index)
        out["gap"] = gap
        out["ac"] = ac
        out["ma"] = ma
        return out

    @staticmethod
    def generate_signals(
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: GapAutocorrParams,
    ) -> SignalFrame:
        n = len(data)
        close = data["close"].to_numpy(dtype=float)
        gap = indicators["gap"].to_numpy(dtype=float)
        ac = indicators["ac"].to_numpy(dtype=float)
        ma = indicators["ma"].to_numpy(dtype=float)

        hold_bars = max(int(params.hold_bars), 1)
        thr = abs(float(params.gap_threshold))
        arm = abs(float(params.ac_arm))

        pos = np.zeros(n, dtype=float)
        regime = 0          # +1 persistence, -1 reversal, 0 neutral
        entry_bar = -10 ** 9
        direction = 0

        for i in range(n):
            a = ac[i]
            if np.isfinite(a):
                if a > arm:
                    regime = 1
                elif a < -arm:
                    regime = -1
                # dead band: hold prior regime (hysteresis)

            held = i - entry_bar
            if held < hold_bars:
                pos[i] = direction
                continue

            # flat and eligible for a new entry
            g = gap[i]
            m = ma[i]
            c = close[i]
            if regime == 0 or not np.isfinite(g) or not np.isfinite(m):
                continue
            if abs(g) < thr:
                continue

            base = 1 if g > 0.0 else -1
            intended = base if regime == 1 else -base

            if intended == 1 and not (c > m):
                continue
            if intended == -1 and not (c < m):
                continue

            entry_bar = i
            direction = intended
            pos[i] = float(intended)

        df = pd.DataFrame(index=data.index)
        df["signal"] = pd.Series(pos, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = 1.0
        return SignalFrame(data=df, signal_column="signal", size_column="size")
