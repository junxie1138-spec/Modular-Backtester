from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class GapInfoRatioParams:
    gap_window: int = 15
    ir_threshold: float = 0.30
    atr_len: int = 14
    trail_k: float = 2.5
    ma_len: int = 200


class GeneratedStrategy(BaseStrategy[GapInfoRatioParams]):
    strategy_id = "gen_a2_1779153395"

    @classmethod
    def params_type(cls):
        return GapInfoRatioParams

    @staticmethod
    def warmup_bars(params: GapInfoRatioParams) -> int:
        return int(max(params.gap_window + 1, params.ma_len, params.atr_len + 1)) + 5

    @staticmethod
    def indicators(data: pd.DataFrame, params: GapInfoRatioParams) -> pd.DataFrame:
        close = data["close"].astype(float)
        high = data["high"].astype(float)
        low = data["low"].astype(float)
        open_ = data["open"].astype(float)

        prev_close = close.shift(1)

        # Overnight gap series (signal). Its rolling mean/std is the SNR.
        gap = open_ / prev_close - 1.0
        gap_mean = gap.rolling(params.gap_window).mean()
        gap_std = gap.rolling(params.gap_window).std()
        gap_std_safe = gap_std.where(gap_std > 1e-12)
        gap_ir = gap_mean / gap_std_safe

        # ATR for the trailing stop and regime MA.
        tr = pd.concat(
            [
                (high - low).abs(),
                (high - prev_close).abs(),
                (low - prev_close).abs(),
            ],
            axis=1,
        ).max(axis=1)
        atr = tr.rolling(params.atr_len).mean()

        ma = close.rolling(params.ma_len).mean()

        out = pd.DataFrame(index=data.index)
        out["gap_ir"] = gap_ir
        out["atr"] = atr
        out["ma"] = ma
        return out

    @staticmethod
    def generate_signals(
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: GapInfoRatioParams,
    ) -> SignalFrame:
        close = data["close"].astype(float).to_numpy()
        gap_ir = indicators["gap_ir"].to_numpy()
        atr = indicators["atr"].to_numpy()
        ma = indicators["ma"].to_numpy()

        n = len(close)
        signal = np.zeros(n, dtype=np.int64)

        thr = float(params.ir_threshold)
        k = float(params.trail_k)

        pos = 0          # 0 flat, 1 long, -1 short
        hwm = 0.0        # highest close since long entry
        lwm = 0.0        # lowest close since short entry

        for i in range(n):
            ir = gap_ir[i]
            a = atr[i]
            m = ma[i]
            c = close[i]

            ready = (
                np.isfinite(ir)
                and np.isfinite(a)
                and a > 0.0
                and np.isfinite(m)
                and np.isfinite(c)
            )
            if not ready:
                pos = 0
                signal[i] = 0
                continue

            if pos == 0:
                if ir > thr and c > m:
                    pos = 1
                    hwm = c
                    signal[i] = 1
                elif ir < -thr and c < m:
                    pos = -1
                    lwm = c
                    signal[i] = -1
                else:
                    signal[i] = 0
            elif pos == 1:
                if c > hwm:
                    hwm = c
                if c <= hwm - k * a:
                    pos = 0
                    signal[i] = 0
                else:
                    signal[i] = 1
            else:  # pos == -1
                if c < lwm:
                    lwm = c
                if c >= lwm + k * a:
                    pos = 0
                    signal[i] = 0
                else:
                    signal[i] = -1

        df = pd.DataFrame(index=data.index)
        df["signal"] = signal
        df["size"] = 1.0
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")
