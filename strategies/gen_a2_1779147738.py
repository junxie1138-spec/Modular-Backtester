from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy

_ATR_PERIOD = 14
_MAX_HOLD = 2


@dataclass(slots=True)
class Params:
    compression_window: int = 10
    atr_mult: float = 2.0


class GeneratedStrategy(BaseStrategy[Params]):
    strategy_id = "gen_a2_1779147738"

    @classmethod
    def params_type(cls):
        return Params

    @staticmethod
    def warmup_bars(params: Params) -> int:
        return int(max(int(params.compression_window), _ATR_PERIOD)) + 1

    @staticmethod
    def indicators(data: pd.DataFrame, params: Params) -> pd.DataFrame:
        high = data["high"].astype(float)
        low = data["low"].astype(float)
        close = data["close"].astype(float)
        prev_close = close.shift(1)

        range_hl = (high - low).clip(lower=0.0)
        win = max(int(params.compression_window), 2)
        range_min = range_hl.rolling(win, min_periods=win).min()

        tr = pd.concat(
            [
                (high - low),
                (high - prev_close).abs(),
                (low - prev_close).abs(),
            ],
            axis=1,
        ).max(axis=1)
        atr = tr.rolling(_ATR_PERIOD, min_periods=_ATR_PERIOD).mean()

        clv = ((close - low) / range_hl.replace(0.0, np.nan)).fillna(0.5)

        ind = pd.DataFrame(index=data.index)
        ind["range_hl"] = range_hl
        ind["range_min"] = range_min
        ind["atr"] = atr
        ind["clv"] = clv
        return ind

    @staticmethod
    def generate_signals(
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: Params,
    ) -> SignalFrame:
        idx = data.index
        n = len(idx)
        close = data["close"].to_numpy(dtype=float)
        atr = indicators["atr"].to_numpy(dtype=float)
        range_hl = indicators["range_hl"].to_numpy(dtype=float)
        range_min = indicators["range_min"].to_numpy(dtype=float)
        clv = indicators["clv"].to_numpy(dtype=float)

        trigger = (
            np.isfinite(range_min)
            & np.isfinite(atr)
            & np.isfinite(range_hl)
            & (range_hl <= range_min + 1e-12)
            & (clv >= 0.5)
        )

        position = np.zeros(n, dtype=int)
        in_pos = False
        stop = 0.0
        held = 0
        k = float(params.atr_mult)

        for i in range(n):
            if in_pos:
                held += 1
                if close[i] <= stop or held >= _MAX_HOLD:
                    in_pos = False
                    position[i] = 0
                else:
                    position[i] = 1
            if not in_pos and bool(trigger[i]):
                in_pos = True
                stop = close[i] - k * atr[i]
                held = 0
                position[i] = 1

        df = pd.DataFrame(index=idx)
        df["signal"] = position
        df["size"] = 1.0
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")
