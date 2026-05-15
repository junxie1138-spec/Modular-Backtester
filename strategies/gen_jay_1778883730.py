from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class Params:
    gap_thresh: float = 0.005
    atr_mult: float = 1.5


class GeneratedStrategy(BaseStrategy["Params"]):
    strategy_id = "gen_jay_1778883730"

    @classmethod
    def params_type(cls):
        return Params

    @staticmethod
    def warmup_bars(params: Params) -> int:
        # ATR(14) needs 14 bars; shift(1) adds 1; refractory rolling(5) adds 4
        return 20

    @staticmethod
    def indicators(data: pd.DataFrame, params: Params) -> pd.DataFrame:
        ind = pd.DataFrame(index=data.index)

        prev_close = data["close"].shift(1)
        ind["gap"] = (data["open"] - prev_close) / prev_close

        high, low = data["high"], data["low"]
        tr = pd.concat(
            [
                high - low,
                (high - prev_close).abs(),
                (low - prev_close).abs(),
            ],
            axis=1,
        ).max(axis=1)
        ind["atr"] = tr.rolling(14, min_periods=14).mean()

        # Spike: any gap (up or down) larger than 2x the entry threshold
        spike = (ind["gap"].abs() > 2.0 * params.gap_thresh).astype(float)
        # Refractory: was there a spike in the last 5 bars?
        ind["recent_spike"] = spike.rolling(5, min_periods=5).max()

        return ind

    @staticmethod
    def generate_signals(
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: Params,
    ) -> SignalFrame:
        close = data["close"].to_numpy()
        gap = indicators["gap"].to_numpy()
        atr = indicators["atr"].to_numpy()
        recent_spike = indicators["recent_spike"].to_numpy()

        n = len(data)
        signal = np.zeros(n, dtype=int)
        size = np.ones(n, dtype=float)

        in_trade = False
        hwm = 0.0

        for i in range(n):
            g = gap[i]
            a = atr[i]
            rs = recent_spike[i]
            c = close[i]

            if np.isnan(g) or np.isnan(a) or np.isnan(rs) or a <= 0.0:
                signal[i] = 0
                if in_trade:
                    in_trade = False
                continue

            if in_trade:
                hwm = max(hwm, c)
                stop = hwm - params.atr_mult * a
                if c <= stop:
                    in_trade = False
                    signal[i] = 0
                else:
                    signal[i] = 1
            else:
                # Enter only on meaningful down-gap with no recent gap spike
                if g < -params.gap_thresh and rs < 0.5:
                    in_trade = True
                    hwm = c
                    signal[i] = 1
                else:
                    signal[i] = 0

        df = pd.DataFrame({"signal": signal, "size": size}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)

        return SignalFrame(data=df, signal_column="signal", size_column="size")
