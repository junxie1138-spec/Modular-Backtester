from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class GeneratedParams:
    roc_window: int = 3
    atr_window: int = 14
    activation_pct: float = 0.004
    trail_atr: float = 1.2
    max_hold: int = 2
    size: float = 1.0


class GeneratedStrategy(BaseStrategy[GeneratedParams]):
    strategy_id = "gen_local_1791224076"

    @classmethod
    def params_type(cls):
        return GeneratedParams

    def warmup_bars(self, params: GeneratedParams) -> int:
        return max(200, params.roc_window + 2, params.atr_window + 1)

    def indicators(self, data: pd.DataFrame, params: GeneratedParams) -> pd.DataFrame:
        close = data["close"]
        prev_close = close.shift(1)
        roc = close.pct_change(params.roc_window)
        tr = pd.concat(
            [data["high"] - data["low"],
             (data["high"] - prev_close).abs(),
             (data["low"] - prev_close).abs()],
            axis=1,
        ).max(axis=1)
        return pd.DataFrame({
            "ma200": close.rolling(200).mean(),
            "accel": roc.diff(),
            "gap": data["open"] / prev_close - 1.0,
            "atr": tr.rolling(params.atr_window).mean(),
        }, index=data.index)

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: GeneratedParams,
    ) -> SignalFrame:
        opened = data["open"].to_numpy(dtype=float)
        close = data["close"].to_numpy(dtype=float)
        ma = indicators["ma200"].to_numpy(dtype=float)
        accel = indicators["accel"].to_numpy(dtype=float)
        gap = indicators["gap"].to_numpy(dtype=float)
        atr = indicators["atr"].to_numpy(dtype=float)
        raw = np.zeros(len(data), dtype=int)
        direction = 0
        entry = 0.0
        stop = 0.0
        armed = False
        held = 0

        for i in range(len(data)):
            if direction:
                if held == 0:
                    entry = opened[i]
                    stop = entry - direction * params.trail_atr * atr[i - 1]
                held += 1
                if direction * (close[i] / entry - 1.0) >= params.activation_pct:
                    armed = True
                if armed:
                    candidate = close[i] - direction * params.trail_atr * atr[i]
                    stop = max(stop, entry, candidate) if direction == 1 else min(stop, entry, candidate)
                hit = close[i] <= stop if direction == 1 else close[i] >= stop
                if hit or held >= params.max_hold:
                    direction = 0
                    armed = False
                    held = 0
                raw[i] = direction
                continue

            if i < self.warmup_bars(params) - 1 or not np.isfinite([ma[i], accel[i], accel[i - 1], gap[i], atr[i]]).all():
                continue
            if close[i] > ma[i] and gap[i] < 0 and close[i] > opened[i] and accel[i - 1] <= 0 < accel[i]:
                direction = 1
            elif close[i] < ma[i] and gap[i] > 0 and close[i] < opened[i] and accel[i - 1] >= 0 > accel[i]:
                direction = -1
            raw[i] = direction

        df = pd.DataFrame({"signal": raw, "size": float(params.size)}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")
