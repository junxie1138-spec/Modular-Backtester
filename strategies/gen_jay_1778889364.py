from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class Params:
    momentum_window: int = 10
    atr_window: int = 14
    rank_window: int = 63
    momentum_threshold: float = 0.70
    atr_threshold: float = 0.45
    hold_bars: int = 2


class GeneratedStrategy(BaseStrategy[Params]):
    strategy_id = "gen_jay_1778889364"

    @classmethod
    def params_type(cls) -> type[Params]:
        return Params

    @staticmethod
    def warmup_bars(params: Params) -> int:
        return params.rank_window + max(params.momentum_window, params.atr_window) + 1

    @staticmethod
    def indicators(data: pd.DataFrame, params: Params) -> pd.DataFrame:
        close = data["close"]
        high = data["high"]
        low = data["low"]

        momentum = close.pct_change(params.momentum_window)

        prev_close = close.shift(1)
        tr = pd.concat(
            [
                high - low,
                (high - prev_close).abs(),
                (low - prev_close).abs(),
            ],
            axis=1,
        ).max(axis=1)
        atr = tr.rolling(params.atr_window, min_periods=params.atr_window).mean()

        momentum_rank = momentum.rolling(
            params.rank_window, min_periods=params.rank_window
        ).rank(pct=True)
        atr_rank = atr.rolling(
            params.rank_window, min_periods=params.rank_window
        ).rank(pct=True)

        ind = pd.DataFrame(index=data.index)
        ind["momentum_rank"] = momentum_rank
        ind["atr_rank"] = atr_rank
        return ind

    @staticmethod
    def generate_signals(
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: Params,
    ) -> SignalFrame:
        n = len(data)
        signal = np.zeros(n, dtype=int)
        size = np.ones(n, dtype=float)

        momentum_rank = indicators["momentum_rank"].values
        atr_rank = indicators["atr_rank"].values

        in_position = False
        bars_held = 0
        entry_size = 1.0

        for i in range(n):
            mr = momentum_rank[i]
            ar = atr_rank[i]

            if in_position:
                bars_held += 1
                if bars_held >= params.hold_bars:
                    signal[i] = 0
                    size[i] = 1.0
                    in_position = False
                    bars_held = 0
                else:
                    signal[i] = 1
                    size[i] = entry_size
            else:
                if (
                    not np.isnan(mr)
                    and not np.isnan(ar)
                    and mr >= params.momentum_threshold
                    and ar <= params.atr_threshold
                ):
                    entry_size = 0.5 + 0.5 * float(mr)
                    signal[i] = 1
                    size[i] = entry_size
                    in_position = True
                    bars_held = 0
                else:
                    signal[i] = 0
                    size[i] = 1.0

        df = pd.DataFrame({"signal": signal, "size": size}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = df["size"].shift(1).fillna(1.0)

        return SignalFrame(data=df, signal_column="signal", size_column="size")
