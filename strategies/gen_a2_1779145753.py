from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class StreakRegimeParams:
    streak_len: int = 3
    regime_sma: int = 100
    regime_slope: int = 20
    atr_period: int = 14
    atr_mult: float = 3.0


class GeneratedStrategy(BaseStrategy[StreakRegimeParams]):
    strategy_id = "gen_a2_1779145753"

    @classmethod
    def params_type(cls) -> type[StreakRegimeParams]:
        return StreakRegimeParams

    def warmup_bars(self, params: StreakRegimeParams) -> int:
        return int(max(params.regime_sma + params.regime_slope, params.atr_period) + 1)

    def indicators(self, data: pd.DataFrame, params: StreakRegimeParams) -> pd.DataFrame:
        close = data["close"]
        high = data["high"]
        low = data["low"]
        prev_close = close.shift(1)

        # ATR (NaN-safe true range; first bar falls back to high-low).
        tr = pd.concat(
            [
                (high - low),
                (high - prev_close).abs(),
                (low - prev_close).abs(),
            ],
            axis=1,
        ).max(axis=1)
        atr = tr.rolling(params.atr_period, min_periods=params.atr_period).mean()

        # Primitive 1: consecutive up-close streak count.
        up = (close > prev_close).fillna(False)
        grp = (~up).cumsum()
        streak = up.groupby(grp).cumsum().astype(float)

        # Primitive 2: plastic regime - price above a rising long SMA.
        sma = close.rolling(params.regime_sma, min_periods=params.regime_sma).mean()
        regime_ok = (
            (close > sma) & (sma > sma.shift(params.regime_slope))
        ).fillna(False).astype(int)

        out = pd.DataFrame(index=data.index)
        out["atr"] = atr
        out["streak"] = streak
        out["sma"] = sma
        out["regime_ok"] = regime_ok
        return out

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: StreakRegimeParams,
    ) -> SignalFrame:
        close = data["close"].to_numpy(dtype=float)
        atr = indicators["atr"].to_numpy(dtype=float)
        streak = indicators["streak"].to_numpy(dtype=float)
        regime = indicators["regime_ok"].to_numpy(dtype=float)
        n = len(close)

        sig = np.zeros(n, dtype=int)
        start = self.warmup_bars(params)

        in_pos = False
        hwm = 0.0  # highest close since entry (ratchets up only)

        for i in range(n):
            if i < start or not np.isfinite(atr[i]):
                continue
            if in_pos:
                if close[i] > hwm:
                    hwm = close[i]
                stop = hwm - params.atr_mult * atr[i]
                if close[i] < stop:
                    in_pos = False
                    sig[i] = 0
                else:
                    sig[i] = 1
            else:
                # Two-primitive AND: streak must agree with the plastic regime.
                if streak[i] >= params.streak_len and regime[i] > 0.0:
                    in_pos = True
                    hwm = close[i]
                    sig[i] = 1

        df = pd.DataFrame(index=data.index)
        df["signal"] = sig
        df["size"] = 1.0
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")
