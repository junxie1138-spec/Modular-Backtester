from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class TideTurnParams:
    tide_len: int = 50
    regime_len: int = 200
    atr_len: int = 14
    min_streak: int = 5
    stop_k: float = 2.5
    max_hold: int = 2


class GeneratedStrategy(BaseStrategy[TideTurnParams]):
    strategy_id = "gen_a1_1778912796"

    @classmethod
    def params_type(cls):
        return TideTurnParams

    @staticmethod
    def warmup_bars(params: TideTurnParams) -> int:
        return int(max(params.tide_len, params.regime_len, params.atr_len + 1))

    @staticmethod
    def indicators(data: pd.DataFrame, params: TideTurnParams) -> pd.DataFrame:
        close = data["close"]
        high = data["high"]
        low = data["low"]

        # Tide level: the mean water line price oscillates around.
        tide = close.rolling(params.tide_len).mean()
        regime_ma = close.rolling(params.regime_len).mean()

        prev_close = close.shift(1)
        tr = pd.concat(
            [
                high - low,
                (high - prev_close).abs(),
                (low - prev_close).abs(),
            ],
            axis=1,
        ).max(axis=1)
        atr = tr.rolling(params.atr_len).mean()

        # Consecutive count of bars submerged below the tide level.
        below = close < tide
        group = (~below).cumsum()
        streak = below.groupby(group).cumsum().astype(float)

        up_close = (close > prev_close).astype(float)

        ind = pd.DataFrame(index=data.index)
        ind["tide"] = tide
        ind["regime_ma"] = regime_ma
        ind["atr"] = atr
        ind["streak"] = streak
        ind["up_close"] = up_close
        return ind

    @staticmethod
    def generate_signals(
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: TideTurnParams,
    ) -> SignalFrame:
        close = data["close"].to_numpy(dtype=float)
        streak = indicators["streak"].to_numpy(dtype=float)
        up_close = indicators["up_close"].to_numpy(dtype=float)
        regime_ma = indicators["regime_ma"].to_numpy(dtype=float)
        atr = indicators["atr"].to_numpy(dtype=float)

        n = len(close)
        raw = np.zeros(n, dtype=int)
        warmup = GeneratedStrategy.warmup_bars(params)

        pos = 0
        entry_price = 0.0
        entry_atr = 0.0
        bars_held = 0

        for i in range(n):
            if i < warmup or i < 2:
                raw[i] = pos
                continue

            if pos == 0:
                # Tide was out: a long submerged streak as of two bars ago,
                # before the confirmation window.
                tide_was_out = (
                    np.isfinite(streak[i - 2])
                    and streak[i - 2] >= params.min_streak
                )
                # Two-bar confirmation: the last two bars both closed up.
                two_bar_up = up_close[i] > 0.5 and up_close[i - 1] > 0.5
                regime_ok = (
                    np.isfinite(regime_ma[i]) and close[i] > regime_ma[i]
                )
                atr_ok = np.isfinite(atr[i]) and atr[i] > 0.0
                if tide_was_out and two_bar_up and regime_ok and atr_ok:
                    pos = 1
                    entry_price = close[i]
                    entry_atr = atr[i]
                    bars_held = 0
            else:
                bars_held += 1
                # Fixed volatility stop anchored to entry (not trailing).
                stop_level = entry_price - params.stop_k * entry_atr
                if close[i] < stop_level or bars_held >= params.max_hold:
                    pos = 0

            raw[i] = pos

        df = pd.DataFrame(index=data.index)
        df["signal"] = raw
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = 1.0
        return SignalFrame(data=df, signal_column="signal", size_column="size")
