from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class ReturnMassParams:
    window: int = 8
    p_hi: float = 0.62
    p_lo: float = 0.38
    profit_target: float = 0.05
    time_stop: int = 18
    min_size: float = 0.25


class GeneratedStrategy(BaseStrategy[ReturnMassParams]):
    strategy_id = "gen_a2_1779151062"

    @classmethod
    def params_type(cls):
        return ReturnMassParams

    @staticmethod
    def warmup_bars(params: ReturnMassParams) -> int:
        return int(max(int(params.window), 2)) + 1

    def indicators(self, data: pd.DataFrame, params: ReturnMassParams) -> pd.DataFrame:
        w = int(max(int(params.window), 2))
        ret = data["close"].pct_change()
        up = ret.clip(lower=0.0)
        dn = (-ret).clip(lower=0.0)
        up_mass = up.rolling(w, min_periods=w).sum()
        dn_mass = dn.rolling(w, min_periods=w).sum()
        total = up_mass + dn_mass
        pos = up_mass / total
        # total == 0.0 (flat window) -> neutral 0.5; NaN warmup stays NaN.
        pos = pos.mask(total == 0.0, 0.5)
        ind = pd.DataFrame(index=data.index)
        ind["return_mass_pos"] = pos
        return ind

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: ReturnMassParams,
    ) -> SignalFrame:
        close = data["close"].to_numpy(dtype=float)
        pos = indicators["return_mass_pos"].to_numpy(dtype=float)
        n = len(close)

        raw = np.zeros(n, dtype=int)
        size = np.ones(n, dtype=float)

        p_hi = float(params.p_hi)
        p_lo = float(params.p_lo)
        target = float(params.profit_target)
        tstop = int(params.time_stop)
        min_size = float(params.min_size)

        in_pos = 0
        entry_price = 0.0
        entry_bar = 0
        entry_size = 1.0

        for i in range(1, n):
            p = pos[i]
            pp = pos[i - 1]

            if in_pos == 0:
                if np.isfinite(p) and np.isfinite(pp):
                    if pp < p_hi <= p:
                        in_pos = 1
                    elif pp > p_lo >= p:
                        in_pos = -1
                    if in_pos != 0:
                        entry_price = close[i]
                        entry_bar = i
                        conv = abs(p - 0.5) / 0.5
                        entry_size = float(np.clip(conv, min_size, 1.0))
                raw[i] = in_pos
                size[i] = entry_size if in_pos != 0 else 1.0
            else:
                held = i - entry_bar
                if in_pos == 1:
                    gain = close[i] / entry_price - 1.0
                else:
                    gain = entry_price / close[i] - 1.0
                if (gain >= target) or (held >= tstop):
                    raw[i] = 0
                    size[i] = 1.0
                    in_pos = 0
                    entry_size = 1.0
                else:
                    raw[i] = in_pos
                    size[i] = entry_size

        df = pd.DataFrame(index=data.index)
        df["signal"] = pd.Series(raw, index=data.index).shift(1).fillna(0).astype(int)
        size_s = pd.Series(size, index=data.index).astype(float)
        size_s = size_s.where(size_s > 0.0, 1.0)
        df["size"] = size_s
        return SignalFrame(data=df, signal_column="signal", size_column="size")
