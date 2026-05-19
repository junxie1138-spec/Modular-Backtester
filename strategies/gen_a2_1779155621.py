from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class Params:
    vol_window: int = 20
    pct_window: int = 120
    low_pct: float = 0.30
    high_pct: float = 0.80
    queue_capacity: int = 8
    atr_window: int = 14
    stop_k: float = 2.5
    max_hold: int = 5
    target_vol: float = 0.01
    max_size: float = 2.0
    min_size: float = 0.3


class GeneratedStrategy(BaseStrategy[Params]):
    strategy_id = "gen_a2_1779155621"

    @classmethod
    def params_type(cls) -> type[Params]:
        return Params

    @staticmethod
    def warmup_bars(params: Params) -> int:
        return int(params.vol_window + params.pct_window + 1)

    def indicators(self, data: pd.DataFrame, params: Params) -> pd.DataFrame:
        close = data["close"]
        high = data["high"]
        low = data["low"]

        ret = close.pct_change()
        rvol = ret.rolling(params.vol_window).std()
        vol_pct = rvol.rolling(params.pct_window).rank(pct=True)

        prev_close = close.shift(1)
        tr = pd.concat(
            [
                (high - low),
                (high - prev_close).abs(),
                (low - prev_close).abs(),
            ],
            axis=1,
        ).max(axis=1)
        atr = tr.rolling(params.atr_window).mean()

        out = pd.DataFrame(index=data.index)
        out["ret"] = ret
        out["rvol"] = rvol
        out["vol_pct"] = vol_pct
        out["atr"] = atr
        return out

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: Params,
    ) -> SignalFrame:
        close = data["close"].to_numpy(dtype=float)
        vol_pct = indicators["vol_pct"].to_numpy(dtype=float)
        atr = indicators["atr"].to_numpy(dtype=float)
        n = len(close)

        capacity = float(max(1, params.queue_capacity))
        low_pct = float(params.low_pct)
        high_pct = float(params.high_pct)
        stop_k = float(params.stop_k)
        max_hold = int(max(1, params.max_hold))

        pos = np.zeros(n, dtype=float)
        queue = 0.0
        prev_overflow = False
        in_pos = False
        stop_level = 0.0
        held = 0

        for i in range(n):
            vp = vol_pct[i]
            if np.isnan(vp):
                queue = 0.0
            else:
                if vp <= low_pct:
                    queue += 1.0
                elif vp >= high_pct:
                    queue = 0.0
                else:
                    queue = max(0.0, queue - 1.0)
                if queue > capacity:
                    queue = capacity

            overflow = queue >= capacity
            fresh_overflow = overflow and not prev_overflow
            prev_overflow = overflow

            if not in_pos:
                a = atr[i]
                if fresh_overflow and not np.isnan(a) and a > 0.0:
                    in_pos = True
                    stop_level = close[i] - stop_k * a
                    held = 0
                    pos[i] = 1.0
                else:
                    pos[i] = 0.0
            else:
                held += 1
                if close[i] <= stop_level:
                    in_pos = False
                    pos[i] = 0.0
                elif held >= max_hold:
                    in_pos = False
                    pos[i] = 0.0
                else:
                    pos[i] = 1.0

        rvol = indicators["rvol"].replace(0.0, np.nan)
        size = (params.target_vol / rvol).clip(
            lower=params.min_size, upper=params.max_size
        )
        size = size.fillna(params.min_size).astype(float)

        df = pd.DataFrame(index=data.index)
        df["signal"] = pd.Series(pos, index=data.index).shift(1).fillna(0).astype(int)
        df["size"] = size
        return SignalFrame(data=df, signal_column="signal", size_column="size")
