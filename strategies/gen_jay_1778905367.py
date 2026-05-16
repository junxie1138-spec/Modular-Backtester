from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class Params:
    lookback: int = 5
    profit_target_pct: float = 0.015


class GeneratedStrategy(BaseStrategy[Params]):
    strategy_id = "gen_jay_1778905367"

    @classmethod
    def params_type(cls) -> type[Params]:
        return Params

    @staticmethod
    def warmup_bars(params: Params) -> int:
        return 14 + params.lookback + 5

    @staticmethod
    def indicators(data: pd.DataFrame, params: Params) -> pd.DataFrame:
        high = data["high"]
        low = data["low"]
        close = data["close"]

        prev_close = close.shift(1)
        tr = pd.concat(
            [
                high - low,
                (high - prev_close).abs(),
                (low - prev_close).abs(),
            ],
            axis=1,
        ).max(axis=1)
        atr = tr.rolling(14, min_periods=14).mean()

        atr_declining = (atr < atr.shift(1)).fillna(False)
        not_declining = (~atr_declining).astype(int)
        group_id = not_declining.cumsum()
        descent_count = atr_declining.astype(int).groupby(group_id).cumsum()

        ind = pd.DataFrame(index=data.index)
        ind["atr"] = atr
        ind["atr_declining"] = atr_declining.astype(int)
        ind["descent_count"] = descent_count.astype(float)
        return ind

    @staticmethod
    def generate_signals(
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: Params,
    ) -> SignalFrame:
        close = data["close"].to_numpy()
        atr_arr = indicators["atr"].to_numpy()
        atr_dec = indicators["atr_declining"].to_numpy()
        desc = indicators["descent_count"].to_numpy()

        n = len(data)
        signal = np.zeros(n, dtype=int)
        size = np.ones(n, dtype=float)

        TIME_STOP = 2

        in_position = False
        entry_price = 0.0
        bars_held = 0

        for i in range(1, n):
            if in_position:
                bars_held += 1
                ret = (close[i] - entry_price) / entry_price
                if ret >= params.profit_target_pct or bars_held >= TIME_STOP:
                    in_position = False
                    signal[i] = 0
                else:
                    signal[i] = 1
            else:
                if (
                    not np.isnan(atr_arr[i])
                    and atr_dec[i] == 0
                    and desc[i - 1] >= params.lookback
                ):
                    signal[i] = 1
                    in_position = True
                    entry_price = close[i]
                    bars_held = 0

        df = pd.DataFrame({"signal": signal, "size": size}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")
