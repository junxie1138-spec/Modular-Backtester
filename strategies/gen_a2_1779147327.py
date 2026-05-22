from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy

_ATR_WINDOW = 14
_MAX_HOLD = 2


@dataclass(slots=True)
class Params:
    streak_len: int = 3
    atr_mult: float = 1.5


class GeneratedStrategy(BaseStrategy[Params]):
    strategy_id = "gen_a2_1779147327"

    @classmethod
    def params_type(cls) -> type[Params]:
        return Params

    @staticmethod
    def warmup_bars(params: Params) -> int:
        # ATR(14) needs a prior close (+1) plus a buffer for the streak run-up.
        return _ATR_WINDOW + 16

    @staticmethod
    def indicators(data: pd.DataFrame, params: Params) -> pd.DataFrame:
        high = data["high"]
        low = data["low"]
        close = data["close"]
        prev_close = close.shift(1)
        prev_high = high.shift(1)

        true_range = pd.concat(
            [
                high - low,
                (high - prev_close).abs(),
                (low - prev_close).abs(),
            ],
            axis=1,
        ).max(axis=1)
        atr = true_range.rolling(_ATR_WINDOW, min_periods=_ATR_WINDOW).mean()

        # 'High-tide' bar: the close clears the entire prior bar's high.
        tide = (close > prev_high).fillna(False).astype(int)
        # Consecutive run length of high-tide bars (resets on any non-tide bar).
        reset = (tide == 0).cumsum()
        streak = tide.groupby(reset).cumsum()

        out = pd.DataFrame(index=data.index)
        out["atr"] = atr
        out["streak"] = streak.astype(float)
        return out

    @staticmethod
    def generate_signals(
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: Params,
    ) -> SignalFrame:
        close = data["close"].to_numpy(dtype=float)
        atr = indicators["atr"].to_numpy(dtype=float)
        streak = indicators["streak"].to_numpy(dtype=float)
        n = len(close)

        raw = np.zeros(n, dtype=np.int64)
        warmup = GeneratedStrategy.warmup_bars(params)
        thresh = max(1, int(params.streak_len))
        k = float(params.atr_mult)

        position = 0
        stop_level = 0.0
        bars_in_pos = 0

        for i in range(n):
            if i < warmup or not np.isfinite(atr[i]) or not np.isfinite(streak[i]):
                raw[i] = 0
                continue

            if position == 0:
                # Fire exactly when the run hits the threshold (one entry per run).
                if streak[i] == thresh:
                    position = 1
                    bars_in_pos = 0
                    stop_level = close[i] - k * atr[i]
                    raw[i] = 1
                else:
                    raw[i] = 0
            else:
                bars_in_pos += 1
                # Fixed volatility stop: close below entry-anchored level, or time cap.
                if close[i] <= stop_level or bars_in_pos >= _MAX_HOLD:
                    position = 0
                    bars_in_pos = 0
                    stop_level = 0.0
                    raw[i] = 0
                else:
                    raw[i] = 1

        # Mandatory one-bar shift: decide on bar N's close, fill on N+1.
        signal = pd.Series(raw, index=data.index).shift(1).fillna(0).astype(int)
        size = pd.Series(1.0, index=data.index)

        df = pd.DataFrame(index=data.index)
        df["signal"] = signal
        df["size"] = size
        return SignalFrame(data=df, signal_column="signal", size_column="size")
