from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class Params:
    recovery_window: int = 20
    momentum_window: int = 10
    rank_lookback: int = 60
    rank_threshold: float = 0.65
    atr_window: int = 14
    atr_stop_mult: float = 2.0


class GeneratedStrategy(BaseStrategy["Params"]):
    strategy_id = "gen_jay_1778903403"

    @classmethod
    def params_type(cls):
        return Params

    @staticmethod
    def warmup_bars(params: Params) -> int:
        return (
            params.rank_lookback
            + max(params.recovery_window, params.momentum_window)
            + params.atr_window
            + 2
        )

    @staticmethod
    def indicators(data: pd.DataFrame, params: Params) -> pd.DataFrame:
        ind = pd.DataFrame(index=data.index)
        high = data["high"]
        low = data["low"]
        close = data["close"]

        bar_range = (high - low).replace(0, np.nan)
        recovery = ((close - low) / bar_range).fillna(0.5)

        rolling_rec = recovery.rolling(
            params.recovery_window, min_periods=params.recovery_window
        ).mean()

        momentum = close.pct_change(params.momentum_window)

        rec_threshold = rolling_rec.rolling(
            params.rank_lookback, min_periods=params.rank_lookback
        ).quantile(params.rank_threshold)
        mom_threshold = momentum.rolling(
            params.rank_lookback, min_periods=params.rank_lookback
        ).quantile(params.rank_threshold)

        ind["recovery_high"] = (rolling_rec > rec_threshold).astype(float)
        ind["momentum_high"] = (momentum > mom_threshold).astype(float)

        tr1 = high - low
        tr2 = (high - close.shift(1)).abs()
        tr3 = (low - close.shift(1)).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        ind["atr"] = tr.rolling(params.atr_window, min_periods=params.atr_window).mean()

        return ind

    @staticmethod
    def generate_signals(
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: Params,
    ) -> SignalFrame:
        close_arr = data["close"].values
        recovery_high = indicators["recovery_high"].values
        momentum_high = indicators["momentum_high"].values
        atr_arr = indicators["atr"].values
        n = len(close_arr)

        raw_signal = np.zeros(n, dtype=np.int64)
        in_position = False
        entry_price = 0.0
        entry_atr = 0.0

        for i in range(n):
            if np.isnan(atr_arr[i]):
                raw_signal[i] = 0
                continue

            if in_position:
                stop_level = entry_price - params.atr_stop_mult * entry_atr
                if close_arr[i] <= stop_level:
                    raw_signal[i] = 0
                    in_position = False
                else:
                    raw_signal[i] = 1
            else:
                if recovery_high[i] == 1.0 and momentum_high[i] == 1.0:
                    raw_signal[i] = 1
                    in_position = True
                    entry_price = close_arr[i]
                    entry_atr = atr_arr[i]
                else:
                    raw_signal[i] = 0

        df = pd.DataFrame(index=data.index)
        df["signal"] = raw_signal
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = 1.0

        return SignalFrame(data=df, signal_column="signal", size_column="size")
