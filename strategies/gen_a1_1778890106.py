from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class MomentumTermStructureParams:
    horizon_a: int = 5
    horizon_b: int = 10
    horizon_c: int = 21
    horizon_d: int = 42
    rank_window: int = 63
    entry_thresh: float = 0.70
    exit_thresh: float = 0.30
    profit_target: float = 0.04
    time_stop: int = 10


class GeneratedStrategy(BaseStrategy[MomentumTermStructureParams]):
    """Long/short on cross-horizon momentum rank agreement.

    Momentum is measured at four lookbacks. Each momentum series is converted
    to its rolling percentile rank within its own trailing window. The mean of
    the four ranks is a term-structure agreement composite in [0, 1]. A long
    setup requires the composite above ``entry_thresh`` on two consecutive
    bars; a short setup requires it below ``exit_thresh`` on two consecutive
    bars. Open positions exit at a fixed profit target or after a time stop,
    whichever fires first.
    """

    strategy_id = "gen_a1_1778890106"

    @classmethod
    def params_type(cls):
        return MomentumTermStructureParams

    def warmup_bars(self, params):
        return int(params.horizon_d) + int(params.rank_window) + 2

    def indicators(self, data, params):
        close = data["close"]
        out = pd.DataFrame(index=data.index)
        horizons = [
            max(int(params.horizon_a), 1),
            max(int(params.horizon_b), 1),
            max(int(params.horizon_c), 1),
            max(int(params.horizon_d), 1),
        ]
        win = max(int(params.rank_window), 2)
        rank_cols = []
        for idx, h in enumerate(horizons):
            mom = close.pct_change(h)
            rk = mom.rolling(win, min_periods=win).rank(pct=True)
            col = f"rank_{idx}"
            out[col] = rk
            rank_cols.append(col)
        comp = out[rank_cols].mean(axis=1)
        valid = out[rank_cols].notna().all(axis=1)
        out["composite"] = comp.where(valid)
        return out

    def generate_signals(self, data, indicators, ctx, params):
        close = data["close"].to_numpy(dtype=float)
        comp = indicators["composite"]

        long_raw = (comp > float(params.entry_thresh)).fillna(False)
        short_raw = (comp < float(params.exit_thresh)).fillna(False)
        long_conf = (
            long_raw & long_raw.shift(1, fill_value=False)
        ).to_numpy()
        short_conf = (
            short_raw & short_raw.shift(1, fill_value=False)
        ).to_numpy()

        n = len(close)
        sig = np.zeros(n, dtype=np.int64)
        pos = 0
        entry_price = 0.0
        held = 0
        target = float(params.profit_target)
        max_hold = max(int(params.time_stop), 1)

        for i in range(n):
            if pos == 0:
                if long_conf[i]:
                    pos = 1
                    entry_price = close[i]
                    held = 0
                elif short_conf[i]:
                    pos = -1
                    entry_price = close[i]
                    held = 0
            else:
                held += 1
                if entry_price > 0.0:
                    if pos == 1:
                        pnl = (close[i] - entry_price) / entry_price
                    else:
                        pnl = (entry_price - close[i]) / entry_price
                else:
                    pnl = 0.0
                if pnl >= target or held >= max_hold:
                    pos = 0
                    entry_price = 0.0
                    held = 0
            sig[i] = pos

        df = pd.DataFrame(index=data.index)
        df["signal"] = sig
        df["size"] = 1.0
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")
