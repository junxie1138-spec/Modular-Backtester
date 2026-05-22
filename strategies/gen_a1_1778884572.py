from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class PlasticYieldParams:
    ret_window: int = 15
    rank_window: int = 252
    upper_pct: float = 0.85
    lower_pct: float = 0.15
    hold_bars: int = 18
    atr_window: int = 14
    atr_mult: float = 2.5
    use_short: bool = True


class GeneratedStrategy(BaseStrategy[PlasticYieldParams]):
    """Plastic-yield momentum on the percentile rank of trailing close-to-close returns.

    The trailing N-bar cumulative return is the 'strain'. Its percentile rank
    within a long rolling window is the 'deformation'. When the strain breaches
    an extreme percentile (the adaptive yield-point) the move is treated as
    plastic and ridden in its direction for a few weeks, subject to a fixed
    ATR volatility stop measured from the entry bar.
    """

    strategy_id = "gen_a1_1778884572"

    @classmethod
    def params_type(cls) -> type[PlasticYieldParams]:
        return PlasticYieldParams

    def warmup_bars(self, params: PlasticYieldParams) -> int:
        strain_lb = int(params.ret_window) + int(params.rank_window)
        atr_lb = int(params.atr_window) + 1
        return max(strain_lb, atr_lb) + 2

    def indicators(self, data: pd.DataFrame, params: PlasticYieldParams) -> pd.DataFrame:
        close = data["close"]
        high = data["high"]
        low = data["low"]

        ret = close.pct_change()
        strain = close / close.shift(int(params.ret_window)) - 1.0
        strain_pct = strain.rolling(
            int(params.rank_window), min_periods=int(params.rank_window)
        ).rank(pct=True)

        prev_close = close.shift(1)
        tr = pd.concat(
            [
                (high - low).abs(),
                (high - prev_close).abs(),
                (low - prev_close).abs(),
            ],
            axis=1,
        ).max(axis=1)
        atr = tr.rolling(int(params.atr_window), min_periods=int(params.atr_window)).mean()

        out = pd.DataFrame(index=data.index)
        out["ret"] = ret
        out["strain"] = strain
        out["strain_pct"] = strain_pct
        out["atr"] = atr
        return out

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: PlasticYieldParams,
    ) -> SignalFrame:
        close = data["close"].to_numpy(dtype=float)
        strain_pct = indicators["strain_pct"].to_numpy(dtype=float)
        atr = indicators["atr"].to_numpy(dtype=float)
        n = len(close)

        position = np.zeros(n, dtype=int)
        in_pos = 0
        entry_price = 0.0
        entry_atr = 0.0
        bars_held = 0

        upper = float(params.upper_pct)
        lower = float(params.lower_pct)
        mult = float(params.atr_mult)
        hold = int(params.hold_bars)
        use_short = bool(params.use_short)

        for i in range(n):
            sp = strain_pct[i]
            a = atr[i]

            if in_pos == 0:
                if np.isnan(sp) or np.isnan(a) or a <= 0.0:
                    position[i] = 0
                    continue
                if sp >= upper:
                    in_pos = 1
                elif use_short and sp <= lower:
                    in_pos = -1
                if in_pos != 0:
                    entry_price = close[i]
                    entry_atr = a
                    bars_held = 0
                position[i] = in_pos
            else:
                bars_held += 1
                exit_now = False
                if in_pos == 1:
                    stop = entry_price - mult * entry_atr
                    if close[i] <= stop:
                        exit_now = True
                else:
                    stop = entry_price + mult * entry_atr
                    if close[i] >= stop:
                        exit_now = True
                if bars_held >= hold:
                    exit_now = True
                if exit_now:
                    in_pos = 0
                    position[i] = 0
                else:
                    position[i] = in_pos

        df = pd.DataFrame(index=data.index)
        df["signal"] = position
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = 1.0
        return SignalFrame(data=df, signal_column="signal", size_column="size")
