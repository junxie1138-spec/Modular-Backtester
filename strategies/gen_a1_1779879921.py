from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class GeneratedParams:
    roc_window: int = 20
    atr_mult: float = 2.5


class GeneratedStrategy(BaseStrategy[GeneratedParams]):
    strategy_id = "gen_a1_1779879921"

    @classmethod
    def params_type(cls):
        return GeneratedParams

    @staticmethod
    def warmup_bars(params: GeneratedParams) -> int:
        w = int(params.roc_window)
        return max(2 * w + 5, 30)

    @staticmethod
    def indicators(data: pd.DataFrame, params: GeneratedParams) -> pd.DataFrame:
        w = int(params.roc_window)
        if w < 2:
            w = 2

        close = data["close"].astype(float)
        high = data["high"].astype(float)
        low = data["low"].astype(float)
        prev_close = close.shift(1)

        tr = pd.concat(
            [
                (high - low).abs(),
                (high - prev_close).abs(),
                (low - prev_close).abs(),
            ],
            axis=1,
        ).max(axis=1)
        atr = tr.rolling(14, min_periods=14).mean()

        ret_w = close.pct_change(w)
        accel = ret_w - ret_w.shift(w)

        log_ret = np.log(close / prev_close)
        noise = log_ret.rolling(w, min_periods=w).std() * np.sqrt(w)
        noise_safe = noise.where(noise > 0.0)

        snr = accel / noise_safe

        out = pd.DataFrame(
            {
                "atr": atr,
                "snr": snr,
            },
            index=data.index,
        )
        return out

    @staticmethod
    def generate_signals(
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: GeneratedParams,
    ) -> SignalFrame:
        close = data["close"].to_numpy(dtype=float)
        atr = indicators["atr"].to_numpy(dtype=float)
        snr = indicators["snr"].to_numpy(dtype=float)
        k = float(params.atr_mult)
        entry_threshold = 1.0

        n = close.shape[0]
        raw = np.zeros(n, dtype=np.int64)
        in_pos = False
        hwm = 0.0

        for i in range(n):
            c = close[i]
            a = atr[i]
            s = snr[i]

            if in_pos:
                if not np.isnan(c) and c > hwm:
                    hwm = c
                stop_armed = (not np.isnan(a)) and (a > 0.0)
                if stop_armed and (not np.isnan(c)) and c <= hwm - k * a:
                    raw[i] = 0
                    in_pos = False
                    hwm = 0.0
                else:
                    raw[i] = 1
            else:
                if (
                    (not np.isnan(s))
                    and (not np.isnan(a))
                    and (not np.isnan(c))
                    and (a > 0.0)
                    and (s > entry_threshold)
                ):
                    raw[i] = 1
                    in_pos = True
                    hwm = c
                else:
                    raw[i] = 0

        df = pd.DataFrame(index=data.index)
        df["signal"] = pd.Series(raw, index=data.index).shift(1).fillna(0).astype(int)
        df["size"] = 1.0
        return SignalFrame(data=df, signal_column="signal", size_column="size")
