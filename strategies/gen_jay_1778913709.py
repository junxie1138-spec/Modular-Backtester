from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class Params:
    aniso_window: int = 20
    aniso_threshold: float = 1.8
    gap_threshold: float = 0.25
    atr_window: int = 14
    trail_mult: float = 2.0


class GeneratedStrategy(BaseStrategy[Params]):
    strategy_id = "gen_jay_1778913709"

    @classmethod
    def params_type(cls):
        return Params

    @staticmethod
    def warmup_bars(params: Params) -> int:
        return max(params.aniso_window, params.atr_window) + 5

    @staticmethod
    def indicators(data: pd.DataFrame, params: Params) -> pd.DataFrame:
        ind = pd.DataFrame(index=data.index)

        close = data["close"]
        open_ = data["open"]
        high = data["high"]
        low = data["low"]
        prev_close = close.shift(1)

        tr = pd.concat([
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ], axis=1).max(axis=1)
        atr = tr.rolling(params.atr_window).mean()
        ind["atr"] = atr

        # Gap in ATR units: positive = up-gap, negative = down-gap
        ind["gap"] = (open_ - prev_close) / atr

        # Directional body anisotropy:
        # up-body = sum of (close - open) when positive over window
        # dn-body = sum of (open - close) when positive over window
        # ratio >> 1: upward plastic deformation; << 1: downward
        body = close - open_
        up_body = body.clip(lower=0).rolling(params.aniso_window).sum()
        dn_body = (-body).clip(lower=0).rolling(params.aniso_window).sum()
        ind["anisotropy"] = up_body / (dn_body + 1e-9)

        return ind

    @staticmethod
    def generate_signals(
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: Params,
    ) -> SignalFrame:
        close = data["close"].values
        atr = indicators["atr"].values
        gap = indicators["gap"].values
        aniso = indicators["anisotropy"].values

        n = len(close)
        raw_signal = np.zeros(n, dtype=int)
        size = np.ones(n, dtype=float)

        thresh = params.aniso_threshold
        gap_th = params.gap_threshold

        # Path-dependent trailing-stop state machine
        position = 0
        high_water = 0.0
        low_water = float("inf")

        for i in range(n):
            if np.isnan(atr[i]) or np.isnan(aniso[i]) or np.isnan(gap[i]):
                # Maintain position through NaN warmup bars
                raw_signal[i] = position
                continue

            cur_close = close[i]
            trail = params.trail_mult * atr[i]

            if position == 0:
                # Primitive 1: anisotropy confirms directional plastic deformation
                # Primitive 2: gap aligns with deformation direction
                # AND: both must agree
                if aniso[i] > thresh and gap[i] > gap_th:
                    raw_signal[i] = 1
                    position = 1
                    high_water = cur_close
                elif aniso[i] < (1.0 / thresh) and gap[i] < -gap_th:
                    raw_signal[i] = -1
                    position = -1
                    low_water = cur_close
                # else: remain flat

            elif position == 1:
                # Ratchet high-water mark up only
                if cur_close > high_water:
                    high_water = cur_close
                if cur_close < high_water - trail:
                    # Trailing stop triggered: exit
                    raw_signal[i] = 0
                    position = 0
                    high_water = 0.0
                else:
                    raw_signal[i] = 1

            else:  # position == -1
                # Ratchet low-water mark down only
                if cur_close < low_water:
                    low_water = cur_close
                if cur_close > low_water + trail:
                    # Trailing stop triggered: exit
                    raw_signal[i] = 0
                    position = 0
                    low_water = float("inf")
                else:
                    raw_signal[i] = -1

        df = pd.DataFrame({"signal": raw_signal, "size": size}, index=data.index)
        # MANDATORY: shift by 1 bar — decision at bar N close, fill at bar N+1 open
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)

        return SignalFrame(data=df, signal_column="signal", size_column="size")
