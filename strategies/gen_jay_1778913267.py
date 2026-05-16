from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class Params:
    lookback: int = 20
    hold_bars: int = 4


class GeneratedStrategy(BaseStrategy[Params]):
    strategy_id = "gen_jay_1778913267"

    @classmethod
    def params_type(cls) -> type[Params]:
        return Params

    @staticmethod
    def warmup_bars(params: Params) -> int:
        # regime SMA uses 5*lookback; overflow_thresh needs 2*lookback on top of lookback
        return params.lookback * 5 + 1

    @staticmethod
    def indicators(data: pd.DataFrame, params: Params) -> pd.DataFrame:
        close = data["close"]
        open_ = data["open"]
        prev_close = close.shift(1)

        # Fractional overnight gap; NaN on first bar
        gap = (open_ / prev_close) - 1.0

        # Only downward gaps fill the sell queue
        down_gap = gap.clip(upper=0.0)

        # Rolling cumulative sell-queue load
        queue_load = down_gap.rolling(params.lookback, min_periods=params.lookback).sum()

        # Self-calibrating overflow threshold: 10th percentile of recent queue history
        overflow_thresh = queue_load.rolling(
            params.lookback * 2, min_periods=params.lookback * 2
        ).quantile(0.10)

        # Bull regime: price above long-term MA derived from same lookback
        regime = (
            close > close.rolling(params.lookback * 5, min_periods=params.lookback * 5).mean()
        ).astype(int)

        return pd.DataFrame(
            {
                "queue_load": queue_load,
                "overflow_thresh": overflow_thresh,
                "regime": regime,
            },
            index=data.index,
        )

    @staticmethod
    def generate_signals(
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: Params,
    ) -> SignalFrame:
        n = len(data)
        signal = pd.Series(0, index=data.index, dtype=int)
        size = pd.Series(1.0, index=data.index, dtype=float)

        queue_load = indicators["queue_load"].to_numpy()
        overflow_thresh = indicators["overflow_thresh"].to_numpy()
        regime = indicators["regime"].to_numpy()

        in_trade = False
        bars_held = 0

        for i in range(n):
            if in_trade:
                signal.iloc[i] = 1
                bars_held += 1
                if bars_held >= params.hold_bars:
                    in_trade = False
                    bars_held = 0
            else:
                ql = queue_load[i]
                ot = overflow_thresh[i]
                reg = regime[i]
                if (
                    not np.isnan(ql)
                    and not np.isnan(ot)
                    and ql < 0.0
                    and ql <= ot
                    and reg == 1
                ):
                    signal.iloc[i] = 1
                    in_trade = True
                    bars_held = 1

        # Mandatory one-bar shift: decision at bar N fills at open of bar N+1
        signal = signal.shift(1).fillna(0).astype(int)

        df = data[["close"]].copy()
        df["signal"] = signal
        df["size"] = size

        return SignalFrame(data=df, signal_column="signal", size_column="size")
