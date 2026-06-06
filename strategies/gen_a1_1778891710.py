from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class GeneratedParams:
    er_window: int = 20
    entry_threshold: float = 0.30
    profit_target: float = 0.04
    time_stop: int = 10
    base_size: float = 0.40
    size_gain: float = 0.60
    max_size: float = 1.00


class GeneratedStrategy(BaseStrategy[GeneratedParams]):
    strategy_id = "gen_a1_1778891710"

    @classmethod
    def params_type(cls):
        return GeneratedParams

    @staticmethod
    def warmup_bars(params: GeneratedParams) -> int:
        return int(params.er_window) + 1

    @staticmethod
    def indicators(data: pd.DataFrame, params: GeneratedParams) -> pd.DataFrame:
        win = max(int(params.er_window), 1)
        close = data["close"]
        high = data["high"]
        low = data["low"]

        # Net directional displacement over the window.
        net_move = close - close.shift(win)

        # Cumulative intrabar high-low range = the 'path length' traversed.
        bar_range = (high - low).clip(lower=0.0)
        path = bar_range.rolling(win).sum()
        path = path.where(path > 0.0, np.nan)

        # Range-path efficiency ratio: progress per unit of intrabar churn.
        efficiency = (net_move / path).clip(lower=-1.0, upper=1.0)

        out = pd.DataFrame(index=data.index)
        out["net_move"] = net_move
        out["path"] = path
        out["efficiency"] = efficiency
        return out

    @staticmethod
    def generate_signals(
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: GeneratedParams,
    ) -> SignalFrame:
        n = len(data)
        close = data["close"].to_numpy(dtype=float)
        eff = indicators["efficiency"].to_numpy(dtype=float)

        thr = float(params.entry_threshold)
        denom = max(1.0 - thr, 1e-9)
        pt = float(params.profit_target)
        tstop = max(int(params.time_stop), 1)
        base = float(params.base_size)
        gain = float(params.size_gain)
        cap = float(params.max_size)

        sig = np.zeros(n, dtype=int)
        size = np.ones(n, dtype=float)

        in_pos = False
        entry_idx = -1
        entry_price = 0.0
        held_size = base

        for i in range(n):
            e = eff[i]
            e_prev = eff[i - 1] if i > 0 else np.nan

            if not in_pos:
                fresh_cross = (
                    np.isfinite(e)
                    and np.isfinite(e_prev)
                    and e >= thr
                    and e_prev < thr
                )
                if fresh_cross:
                    # Signal-scaled size: stronger efficiency -> larger position.
                    strength = (e - thr) / denom
                    strength = min(max(strength, 0.0), 1.0)
                    held_size = min(max(base + gain * strength, 0.05), cap)
                    in_pos = True
                    entry_idx = i
                    entry_price = close[i]
                    sig[i] = 1
                    size[i] = held_size
            else:
                bars_held = i - entry_idx
                ret = (close[i] / entry_price) - 1.0 if entry_price > 0.0 else 0.0
                # Exit: profit-target OR time-stop, whichever fires first.
                if ret >= pt or bars_held >= tstop:
                    sig[i] = 0
                    size[i] = 1.0
                    in_pos = False
                    entry_idx = -1
                else:
                    sig[i] = 1
                    size[i] = held_size

        df = pd.DataFrame(index=data.index)
        df["signal"] = sig
        df["size"] = size
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = df["size"].clip(lower=0.05).astype(float)
        return SignalFrame(data=df, signal_column="signal", size_column="size")
