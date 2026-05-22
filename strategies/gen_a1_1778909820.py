from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class AccelAutocorrParams:
    roc_len: int = 5
    accel_smooth: int = 3
    ac_len: int = 20
    pct_len: int = 120
    pct_threshold: float = 0.80
    profit_target: float = 0.03
    time_stop: int = 4


class GeneratedStrategy(BaseStrategy[AccelAutocorrParams]):
    strategy_id = "gen_a1_1778909820"

    @classmethod
    def params_type(cls):
        return AccelAutocorrParams

    @staticmethod
    def warmup_bars(params: AccelAutocorrParams) -> int:
        return int(
            params.roc_len
            + params.accel_smooth
            + params.ac_len
            + params.pct_len
            + 5
        )

    @staticmethod
    def indicators(data: pd.DataFrame, params: AccelAutocorrParams) -> pd.DataFrame:
        close = data["close"]

        # Rate-of-change and its smoothed acceleration (change in ROC).
        roc = close.pct_change(max(1, params.roc_len))
        roc_s = roc.ewm(span=max(1, params.accel_smooth), adjust=False).mean()
        accel = roc_s.diff()

        # Lag-1 autocorrelation of the acceleration impulse stream.
        ac_len = max(3, params.ac_len)
        ac = accel.rolling(ac_len).corr(accel.shift(1))

        # Percentile rank of that autocorrelation within its own history (the twist).
        pct_len = max(5, params.pct_len)
        ac_rank = ac.rolling(pct_len).rank(pct=True)

        ind = pd.DataFrame(index=data.index)
        ind["roc_s"] = roc_s
        ind["accel"] = accel
        ind["ac"] = ac
        ind["ac_rank"] = ac_rank
        return ind

    @staticmethod
    def generate_signals(
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: AccelAutocorrParams,
    ) -> SignalFrame:
        close = data["close"].to_numpy(dtype=float)
        n = len(close)

        roc_s = indicators["roc_s"].fillna(0.0).to_numpy(dtype=float)
        accel = indicators["accel"].fillna(0.0).to_numpy(dtype=float)
        ac_rank = indicators["ac_rank"].fillna(0.0).to_numpy(dtype=float)

        thr = float(params.pct_threshold)
        pt = float(params.profit_target)
        time_stop = max(1, int(params.time_stop))

        # Entry: acceleration persistence regime is in its top percentile,
        # price is accelerating upward, and momentum itself is positive.
        entry = (ac_rank >= thr) & (accel > 0.0) & (roc_s > 0.0)

        signal = np.zeros(n, dtype=np.int64)
        position = 0
        entry_price = 0.0
        bars_held = 0

        for i in range(n):
            if position == 0:
                if entry[i]:
                    position = 1
                    entry_price = close[i]
                    bars_held = 0
                    signal[i] = 1
                else:
                    signal[i] = 0
            else:
                bars_held += 1
                hit_target = entry_price > 0.0 and close[i] >= entry_price * (1.0 + pt)
                hit_time = bars_held >= time_stop
                if hit_target or hit_time:
                    position = 0
                    entry_price = 0.0
                    bars_held = 0
                    signal[i] = 0
                else:
                    signal[i] = 1

        df = pd.DataFrame(index=data.index)
        df["signal"] = signal
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = 1.0
        return SignalFrame(data=df, signal_column="signal", size_column="size")
