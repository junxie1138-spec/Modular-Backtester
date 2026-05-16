from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class GapAutocorrSpringParams:
    lookback: int = 20
    trail_atr_mult: float = 1.5


class GeneratedStrategy(BaseStrategy["GapAutocorrSpringParams"]):
    strategy_id = "gen_jay_1778889762"

    @classmethod
    def params_type(cls):
        return GapAutocorrSpringParams

    @staticmethod
    def warmup_bars(params: GapAutocorrSpringParams) -> int:
        # gap needs 1 lag; gap_lag1 shifts gap by another bar;
        # rolling corr needs lookback valid pairs => lookback + 2 bars total
        return params.lookback + 3

    @staticmethod
    def indicators(
        data: pd.DataFrame, params: GapAutocorrSpringParams
    ) -> pd.DataFrame:
        ind = pd.DataFrame(index=data.index)

        prev_close = data["close"].shift(1)
        gap = (data["open"] - prev_close) / prev_close.replace(0, np.nan)
        ind["gap"] = gap

        gap_lag1 = gap.shift(1)
        ind["gap_autocorr"] = gap.rolling(params.lookback).corr(gap_lag1)

        tr = pd.concat(
            [
                data["high"] - data["low"],
                (data["high"] - prev_close).abs(),
                (data["low"] - prev_close).abs(),
            ],
            axis=1,
        ).max(axis=1)
        ind["atr"] = tr.rolling(params.lookback).mean()

        return ind

    @staticmethod
    def generate_signals(
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: GapAutocorrSpringParams,
    ) -> SignalFrame:
        n = len(data)
        raw_signal = np.zeros(n, dtype=int)
        size_arr = np.ones(n, dtype=float)

        close = data["close"].values
        high = data["high"].values
        gap = indicators["gap"].values
        gap_autocorr = indicators["gap_autocorr"].values
        atr = indicators["atr"].values

        BREAKEVEN_PCT = 0.003  # 0.3% above entry arms the breakeven stop
        MAX_HOLD = 2

        in_position = False
        entry_price = 0.0
        stop_level = 0.0
        breakeven_reached = False
        bars_held = 0

        for i in range(n):
            if in_position:
                bars_held += 1
                cur_high = high[i]
                cur_close = close[i]
                cur_atr = atr[i] if not np.isnan(atr[i]) else 0.0

                # Breakeven: once price taps +BREAKEVEN_PCT above entry, raise stop to entry
                if (
                    not breakeven_reached
                    and cur_high >= entry_price * (1.0 + BREAKEVEN_PCT)
                ):
                    breakeven_reached = True
                    stop_level = max(stop_level, entry_price)

                # Trail: advance stop upward only
                if cur_atr > 0.0:
                    trail = cur_close - params.trail_atr_mult * cur_atr
                    stop_level = max(stop_level, trail)

                # Exit on stop hit or max hold exhausted
                if cur_close <= stop_level or bars_held >= MAX_HOLD:
                    raw_signal[i] = 0
                    in_position = False
                    entry_price = 0.0
                    stop_level = 0.0
                    breakeven_reached = False
                    bars_held = 0
                else:
                    raw_signal[i] = 1
            else:
                # Entry: down-gap in a negative-autocorrelation (elastic) gap regime
                if (
                    not np.isnan(gap[i])
                    and not np.isnan(gap_autocorr[i])
                    and not np.isnan(atr[i])
                    and atr[i] > 0.0
                    and gap[i] < 0.0
                    and gap_autocorr[i] < 0.0
                ):
                    raw_signal[i] = 1
                    in_position = True
                    entry_price = close[i]
                    stop_level = close[i] - params.trail_atr_mult * atr[i]
                    breakeven_reached = False
                    bars_held = 0
                else:
                    raw_signal[i] = 0

        df = pd.DataFrame(
            {"signal": raw_signal, "size": size_arr},
            index=data.index,
        )
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")
