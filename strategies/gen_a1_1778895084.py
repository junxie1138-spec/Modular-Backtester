from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class OvernightGapMomentumParams:
    gap_window: int = 4
    pctile_window: int = 60
    entry_pct: float = 0.80
    exit_pct: float = 0.45
    trend_len: int = 100
    size_floor: float = 0.6


class GeneratedStrategy(BaseStrategy[OvernightGapMomentumParams]):
    strategy_id = "gen_a1_1778895084"

    @classmethod
    def params_type(cls) -> type[OvernightGapMomentumParams]:
        return OvernightGapMomentumParams

    @staticmethod
    def warmup_bars(params: OvernightGapMomentumParams) -> int:
        reservoir = int(params.gap_window) + int(params.pctile_window)
        return int(max(reservoir, int(params.trend_len))) + 5

    @staticmethod
    def indicators(data: pd.DataFrame, params: OvernightGapMomentumParams) -> pd.DataFrame:
        close = data["close"]
        open_ = data["open"]
        prior_close = close.shift(1)

        gap_w = max(int(params.gap_window), 1)
        pct_w = max(int(params.pctile_window), 2)
        trend_w = max(int(params.trend_len), 1)

        # overnight (gap) return: open versus prior close
        gap_ret = (open_ / prior_close) - 1.0
        gap_ret = gap_ret.replace([np.inf, -np.inf], np.nan)

        # cumulative overnight-momentum reservoir over recent bars
        gap_mom = gap_ret.rolling(gap_w, min_periods=gap_w).sum()

        # trailing percentile rank of the reservoir against its own history
        gap_mom_pctile = gap_mom.rolling(pct_w, min_periods=pct_w).rank(pct=True)

        sma = close.rolling(trend_w, min_periods=trend_w).mean()

        out = pd.DataFrame(index=data.index)
        out["gap_mom"] = gap_mom
        out["gap_mom_pctile"] = gap_mom_pctile
        out["sma"] = sma
        return out

    @staticmethod
    def generate_signals(
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: OvernightGapMomentumParams,
    ) -> SignalFrame:
        close = data["close"].to_numpy(dtype=float)
        pctile = indicators["gap_mom_pctile"].to_numpy(dtype=float)
        sma = indicators["sma"].to_numpy(dtype=float)

        n = len(close)
        signal = np.zeros(n, dtype=int)
        size = np.ones(n, dtype=float)

        entry_pct = float(params.entry_pct)
        exit_pct = float(params.exit_pct)
        # exit band must sit strictly below the entry band for hysteresis
        if exit_pct >= entry_pct:
            exit_pct = max(entry_pct - 0.05, 0.0)
        floor = min(max(float(params.size_floor), 0.05), 1.0)
        denom = max(1.0 - entry_pct, 1e-6)

        pos = 0
        for t in range(n):
            p = pctile[t]
            s = sma[t]
            c = close[t]

            if np.isnan(p) or np.isnan(s):
                pos = 0
                signal[t] = 0
                continue

            entry_ok = (p >= entry_pct) and (c > s)

            if pos == 0:
                if entry_ok:
                    pos = 1
                    signal[t] = 1
                else:
                    signal[t] = 0
            else:
                # signal-reversal exit: hold until the entry condition flips off
                if (p < exit_pct) or (c <= s):
                    pos = 0
                    signal[t] = 0
                else:
                    signal[t] = 1

            if signal[t] == 1:
                # capacity scaling: deeper into the high band -> larger size
                strength = (p - entry_pct) / denom
                if strength < 0.0:
                    strength = 0.0
                elif strength > 1.0:
                    strength = 1.0
                size[t] = floor + (1.0 - floor) * strength

        df = pd.DataFrame(index=data.index)
        df["signal"] = signal
        df["size"] = size
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = df["size"].shift(1).fillna(1.0).astype(float)
        return SignalFrame(data=df, signal_column="signal", size_column="size")
