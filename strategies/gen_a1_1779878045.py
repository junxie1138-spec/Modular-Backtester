from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class GenA1Params:
    capacity_window: int = 10
    expected_window: int = 60
    compression_threshold: float = 0.55
    min_return: float = 0.0015
    atr_window: int = 14
    trail_atr_mult: float = 1.75
    ma_filter: int = 200
    use_ma_filter: bool = False


class GeneratedStrategy(BaseStrategy[GenA1Params]):
    strategy_id = "gen_a1_1779878045"

    @classmethod
    def params_type(cls) -> type[GenA1Params]:
        return GenA1Params

    def warmup_bars(self, params: GenA1Params) -> int:
        return max(
            params.capacity_window + params.expected_window,
            params.atr_window,
            params.ma_filter,
        ) + 2

    def indicators(self, data: pd.DataFrame, params: GenA1Params) -> pd.DataFrame:
        close = data["close"]
        high = data["high"]
        low = data["low"]

        ret = close.pct_change()
        abs_ret = ret.abs()

        capacity_used = abs_ret.rolling(
            params.capacity_window, min_periods=params.capacity_window
        ).sum()
        expected_capacity = capacity_used.rolling(
            params.expected_window, min_periods=params.expected_window
        ).mean()
        compression = capacity_used / expected_capacity.replace(0.0, np.nan)

        prev_close = close.shift(1)
        tr = pd.concat(
            [
                high - low,
                (high - prev_close).abs(),
                (low - prev_close).abs(),
            ],
            axis=1,
        ).max(axis=1)
        atr = tr.rolling(params.atr_window, min_periods=params.atr_window).mean()

        ma = close.rolling(params.ma_filter, min_periods=params.ma_filter).mean()

        ind = pd.DataFrame(index=data.index)
        ind["ret"] = ret
        ind["compression"] = compression
        ind["atr"] = atr
        ind["ma"] = ma
        return ind

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: GenA1Params,
    ) -> SignalFrame:
        close = data["close"].to_numpy(dtype=float)
        ret = indicators["ret"].to_numpy(dtype=float)
        compression = indicators["compression"].to_numpy(dtype=float)
        atr = indicators["atr"].to_numpy(dtype=float)
        ma = indicators["ma"].to_numpy(dtype=float)

        n = len(data)
        raw = np.zeros(n, dtype=np.int64)

        pos = 0
        hwm = 0.0
        lwm = 0.0

        for i in range(n):
            c = close[i]
            a = atr[i]

            if pos == 0:
                if i < 2:
                    continue
                if np.isnan(compression[i]) or np.isnan(a):
                    continue
                if compression[i] >= params.compression_threshold:
                    continue
                r0 = ret[i]
                r1 = ret[i - 1]
                if np.isnan(r0) or np.isnan(r1):
                    continue
                if abs(r0) < params.min_return or abs(r1) < params.min_return:
                    continue
                if params.use_ma_filter and np.isnan(ma[i]):
                    continue
                if r0 > 0.0 and r1 > 0.0:
                    if params.use_ma_filter and c <= ma[i]:
                        continue
                    pos = 1
                    hwm = c
                    raw[i] = 1
                elif r0 < 0.0 and r1 < 0.0:
                    if params.use_ma_filter and c >= ma[i]:
                        continue
                    pos = -1
                    lwm = c
                    raw[i] = -1
            elif pos == 1:
                if c > hwm:
                    hwm = c
                if np.isnan(a):
                    pos = 0
                    raw[i] = 0
                else:
                    stop = hwm - params.trail_atr_mult * a
                    if c <= stop:
                        pos = 0
                        raw[i] = 0
                    else:
                        raw[i] = 1
            else:
                if c < lwm:
                    lwm = c
                if np.isnan(a):
                    pos = 0
                    raw[i] = 0
                else:
                    stop = lwm + params.trail_atr_mult * a
                    if c >= stop:
                        pos = 0
                        raw[i] = 0
                    else:
                        raw[i] = -1

        df = pd.DataFrame(index=data.index)
        df["signal"] = pd.Series(raw, index=data.index).shift(1).fillna(0).astype(int)
        df["size"] = 1.0
        return SignalFrame(data=df, signal_column="signal", size_column="size")
