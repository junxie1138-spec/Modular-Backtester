from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class ElasticPlasticParams:
    max_lookback: int = 252
    atr_window: int = 14
    min_dd: float = 0.03
    yield_dd: float = 0.09
    trail_k: float = 2.5
    max_hold_bars: int = 15


class GeneratedStrategy(BaseStrategy[ElasticPlasticParams]):
    strategy_id = "gen_a2_1779154724"

    @classmethod
    def params_type(cls):
        return ElasticPlasticParams

    @staticmethod
    def warmup_bars(params: ElasticPlasticParams) -> int:
        return int(max(params.max_lookback, params.atr_window + 1)) + 2

    def indicators(self, data: pd.DataFrame, params: ElasticPlasticParams) -> pd.DataFrame:
        close = data["close"].astype(float)
        high = data["high"].astype(float)
        low = data["low"].astype(float)

        prev_close = close.shift(1)
        tr = pd.concat(
            [
                high - low,
                (high - prev_close).abs(),
                (low - prev_close).abs(),
            ],
            axis=1,
        ).max(axis=1)
        atr_w = max(int(params.atr_window), 1)
        atr = tr.rolling(atr_w, min_periods=atr_w).mean()

        look = max(int(params.max_lookback), 1)
        roll_max = close.rolling(look, min_periods=1).max()
        dd = close / roll_max - 1.0

        out = pd.DataFrame(index=data.index)
        out["atr"] = atr
        out["dd"] = dd
        return out

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: ElasticPlasticParams,
    ) -> SignalFrame:
        close = data["close"].to_numpy(dtype=float)
        dd = indicators["dd"].to_numpy(dtype=float)
        atr = indicators["atr"].to_numpy(dtype=float)
        n = len(close)
        pos = np.zeros(n, dtype=np.int64)

        min_dd = float(params.min_dd)
        yield_dd = float(params.yield_dd)
        trail_k = float(params.trail_k)
        max_hold = int(params.max_hold_bars)

        in_pos = 0
        hwm = 0.0
        lwm = 0.0
        held = 0

        for i in range(n):
            c = close[i]
            a = atr[i]
            d = dd[i]

            if in_pos == 0:
                confirmed = (
                    i >= 2
                    and close[i] > close[i - 1]
                    and close[i - 1] > close[i - 2]
                )
                valid = (
                    confirmed
                    and not np.isnan(a)
                    and not np.isnan(d)
                    and a > 0.0
                )
                if valid:
                    in_drawdown = d <= -min_dd
                    elastic = in_drawdown and d >= -yield_dd
                    plastic = d < -yield_dd
                    if elastic:
                        in_pos = 1
                        hwm = c
                        held = 0
                    elif plastic:
                        in_pos = -1
                        lwm = c
                        held = 0
            else:
                held += 1
                if in_pos == 1:
                    if c > hwm:
                        hwm = c
                    stop = hwm - trail_k * a
                    if c <= stop or held >= max_hold:
                        in_pos = 0
                else:
                    if c < lwm:
                        lwm = c
                    stop = lwm + trail_k * a
                    if c >= stop or held >= max_hold:
                        in_pos = 0

            pos[i] = in_pos

        df = pd.DataFrame(index=data.index)
        df["signal"] = pos
        df["size"] = 1.0
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")
