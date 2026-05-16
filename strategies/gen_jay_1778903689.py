from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class Params:
    window: int = 10
    atr_mult: float = 2.0


class GeneratedStrategy(BaseStrategy[Params]):
    strategy_id = "gen_jay_1778903689"

    @classmethod
    def params_type(cls):
        return Params

    @staticmethod
    def warmup_bars(params: Params) -> int:
        # gap_pressure: window bars; two diffs add 2; ATR rolling(14) plus shift(1)
        return max(params.window + 3, 17)

    @staticmethod
    def indicators(data: pd.DataFrame, params: Params) -> pd.DataFrame:
        ind = pd.DataFrame(index=data.index)

        gap = data["open"] / data["close"].shift(1) - 1.0
        gap_pressure = gap.rolling(params.window, min_periods=params.window).mean()
        gap_roc = gap_pressure.diff()
        gap_accel = gap_roc.diff()

        prev_close = data["close"].shift(1)
        tr = pd.concat(
            [
                data["high"] - data["low"],
                (data["high"] - prev_close).abs(),
                (data["low"] - prev_close).abs(),
            ],
            axis=1,
        ).max(axis=1)
        atr = tr.rolling(14, min_periods=14).mean()

        ind["gap_pressure"] = gap_pressure
        ind["gap_accel"] = gap_accel
        ind["atr"] = atr

        return ind

    @staticmethod
    def generate_signals(
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: Params,
    ) -> SignalFrame:
        closes = data["close"].to_numpy()
        gap_pressure = indicators["gap_pressure"].to_numpy()
        gap_accel = indicators["gap_accel"].to_numpy()
        atr = indicators["atr"].to_numpy()
        n = len(closes)

        raw_signal = np.zeros(n, dtype=np.int64)
        size = np.ones(n, dtype=np.float64)

        in_trade = False
        entry_price = 0.0
        stop_price = 0.0
        at_breakeven = False

        for i in range(n):
            gp = gap_pressure[i]
            ga = gap_accel[i]
            at = atr[i]
            cl = closes[i]

            if np.isnan(gp) or np.isnan(ga) or np.isnan(at):
                if in_trade:
                    raw_signal[i] = 1
                continue

            if not in_trade:
                # Predator-exhaustion inflection: net negative gap pressure
                # reversing via positive second derivative (hunting rate decelerating)
                if gp < 0.0 and ga > 0.0:
                    raw_signal[i] = 1
                    in_trade = True
                    entry_price = cl
                    stop_price = cl - params.atr_mult * at
                    at_breakeven = False
            else:
                # Breakeven: once price rises 1 ATR from entry, floor stop at entry
                if not at_breakeven and cl >= entry_price + at:
                    stop_price = max(stop_price, entry_price)
                    at_breakeven = True

                # Trail: stop only ever rises, never falls
                trail = cl - params.atr_mult * at
                if trail > stop_price:
                    stop_price = trail

                if cl <= stop_price:
                    raw_signal[i] = 0
                    in_trade = False
                else:
                    raw_signal[i] = 1

        df = pd.DataFrame(
            {"signal": raw_signal, "size": size}, index=data.index
        )
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)

        return SignalFrame(data=df, signal_column="signal", size_column="size")
