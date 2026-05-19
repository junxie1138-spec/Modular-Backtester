from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class SpringTideGapParams:
    window: int = 20
    hold_bars: int = 2


class GeneratedStrategy(BaseStrategy[SpringTideGapParams]):
    """Trade overnight gaps that breach their own rolling amplitude envelope.

    The overnight gap series behaves like a tide: it oscillates around zero
    with a slowly varying amplitude. A gap larger than its recent envelope
    (rolling std) and backed by above-average volume is treated as a genuine
    repricing and traded in the gap's direction, then closed a fixed number
    of bars later regardless of outcome.
    """

    strategy_id = "gen_a2_1779153312"

    @classmethod
    def params_type(cls):
        return SpringTideGapParams

    @staticmethod
    def warmup_bars(params: SpringTideGapParams) -> int:
        return int(params.window) + 1

    @staticmethod
    def indicators(data: pd.DataFrame, params: SpringTideGapParams) -> pd.DataFrame:
        w = max(2, int(params.window))
        prev_close = data["close"].shift(1)
        gap = data["open"] / prev_close.replace(0.0, np.nan) - 1.0
        gap_amp = gap.rolling(w).std()
        vol_mean = data["volume"].rolling(w).mean()

        out = pd.DataFrame(index=data.index)
        out["gap"] = gap
        out["gap_amp"] = gap_amp
        out["vol_mean"] = vol_mean
        return out

    @staticmethod
    def generate_signals(data, indicators, ctx, params) -> SignalFrame:
        gap = indicators["gap"]
        gap_amp = indicators["gap_amp"]
        vol_mean = indicators["vol_mean"]

        # A 'spring tide' gap breaches its own rolling amplitude envelope.
        spring = (gap.abs() > gap_amp).fillna(False)
        vol_conf = (data["volume"] > vol_mean).fillna(False)
        gap_sign = np.sign(gap.fillna(0.0)).astype(int)
        entry = (spring & vol_conf).to_numpy()
        entry_dir = np.where(entry, gap_sign.to_numpy(), 0).astype(int)

        hold = max(1, int(params.hold_bars))
        n = len(data)
        position = np.zeros(n, dtype=int)
        cur = 0
        bars_held = 0
        # Fixed-bar exit: hold exactly `hold` bars after entry, then flatten.
        for i in range(n):
            if cur != 0:
                bars_held += 1
                if bars_held >= hold:
                    cur = 0
                    bars_held = 0
            if cur == 0 and entry_dir[i] != 0:
                cur = int(entry_dir[i])
                bars_held = 0
            position[i] = cur

        df = pd.DataFrame(index=data.index)
        df["signal"] = position
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = 1.0
        return SignalFrame(data=df, signal_column="signal", size_column="size")
