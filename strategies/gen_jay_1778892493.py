from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class Params:
    atr_period: int = 14
    score_period: int = 15
    arm_threshold: float = 1.5
    unarm_threshold: float = 0.5
    atr_stop_mult: float = 2.5


class GeneratedStrategy(BaseStrategy[Params]):
    strategy_id = "gen_jay_1778892493"

    @classmethod
    def params_type(cls) -> type[Params]:
        return Params

    @staticmethod
    def warmup_bars(params: Params) -> int:
        return params.atr_period + params.score_period + 5

    @staticmethod
    def indicators(data: pd.DataFrame, params: Params) -> pd.DataFrame:
        high = data["high"]
        low = data["low"]
        close = data["close"]

        hl = high - low
        hc = (high - close.shift(1)).abs()
        lc = (low - close.shift(1)).abs()
        tr = pd.concat([hl, hc, lc], axis=1).max(axis=1)
        atr = tr.rolling(params.atr_period, min_periods=params.atr_period).mean()

        # ATR-normalized directional range expansion per bar
        up_exp = (high - high.shift(1)).clip(lower=0.0) / atr
        dn_exp = (low.shift(1) - low).clip(lower=0.0) / atr

        # Rolling net directional bias: positive = bulls stretching range harder
        bias_score = (
            (up_exp - dn_exp)
            .rolling(params.score_period, min_periods=params.score_period)
            .sum()
        )

        ind = pd.DataFrame(index=data.index)
        ind["atr"] = atr
        ind["bias_score"] = bias_score
        return ind

    @staticmethod
    def generate_signals(
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: Params,
    ) -> SignalFrame:
        close = data["close"].to_numpy(dtype=float)
        atr = indicators["atr"].to_numpy(dtype=float)
        bias_score = indicators["bias_score"].to_numpy(dtype=float)
        n = len(close)

        raw_signal = np.zeros(n, dtype=int)

        in_trade = False
        entry_price = 0.0
        entry_atr = 0.0
        consecutive_armed = 0

        for i in range(1, n):
            bs = bias_score[i]
            at = atr[i]

            # Hold current state through NaN warmup bars
            if np.isnan(bs) or np.isnan(at) or at <= 0.0:
                raw_signal[i] = 1 if in_trade else 0
                continue

            if in_trade:
                stop_level = entry_price - params.atr_stop_mult * entry_atr
                if close[i] <= stop_level or bs < params.unarm_threshold:
                    # Fixed vol-stop hit, or score fell below hysteresis lower bound
                    in_trade = False
                    entry_price = 0.0
                    entry_atr = 0.0
                    consecutive_armed = 0
                    raw_signal[i] = 0
                else:
                    raw_signal[i] = 1
            else:
                # Two-bar confirmation: must exceed arm_threshold on two consecutive bars
                if bs >= params.arm_threshold:
                    consecutive_armed += 1
                else:
                    consecutive_armed = 0

                if consecutive_armed >= 2:
                    in_trade = True
                    entry_price = close[i]
                    entry_atr = at
                    raw_signal[i] = 1
                else:
                    raw_signal[i] = 0

        df = pd.DataFrame({"signal": raw_signal, "size": 1.0}, index=data.index)
        # Mandatory one-bar shift: decision on bar N fills on bar N+1
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")
