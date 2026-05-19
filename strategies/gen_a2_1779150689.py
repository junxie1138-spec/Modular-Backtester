from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class QueueOverflowParams:
    window: int = 40
    profit_target: float = 0.03


class GeneratedStrategy(BaseStrategy[QueueOverflowParams]):
    strategy_id = "gen_a2_1779150689"

    # Fixed (non-tunable) queue mechanics - keeps the strategy at <=2 tunable params.
    _CAPACITY = 1.0          # hard ceiling of the demand queue
    _SERVICE = 0.5           # neutral drain rate per bar
    _RANK_THRESHOLD = 0.95   # top-percentile overflow trigger
    _TIME_STOP = 5           # exit after at most N held bars

    @classmethod
    def params_type(cls):
        return QueueOverflowParams

    @staticmethod
    def warmup_bars(params: QueueOverflowParams) -> int:
        return int(params.window) + 1

    def indicators(self, data: pd.DataFrame, params: QueueOverflowParams) -> pd.DataFrame:
        high = data["high"].astype(float).to_numpy()
        low = data["low"].astype(float).to_numpy()
        close = data["close"].astype(float).to_numpy()

        rng = high - low
        # Close location value in [0, 1]; flat bars treated as neutral.
        clv = np.where(rng > 0.0, (close - low) / np.where(rng > 0.0, rng, 1.0), 0.5)
        net_flow = clv - self._SERVICE  # arrivals minus service rate

        n = len(data)
        queue = np.empty(n, dtype=float)
        level = self._SERVICE
        cap = self._CAPACITY
        for i in range(n):
            level = level + net_flow[i]
            if level < 0.0:
                level = 0.0
            elif level > cap:
                level = cap
            queue[i] = level

        q = pd.Series(queue, index=data.index)
        window = max(int(params.window), 2)
        qrank = q.rolling(window).rank(pct=True)

        out = pd.DataFrame(index=data.index)
        out["queue"] = q
        out["qrank"] = qrank
        return out

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame,
                         ctx: StrategyContext, params: QueueOverflowParams) -> SignalFrame:
        close = data["close"].to_numpy(dtype=float)
        qrank = indicators["qrank"].to_numpy(dtype=float)
        n = len(data)

        pt = float(params.profit_target)
        thr = self._RANK_THRESHOLD
        time_stop = self._TIME_STOP

        signal = np.zeros(n, dtype=int)
        in_pos = False
        entry_price = 0.0
        bars_held = 0

        for i in range(n):
            trigger = np.isfinite(qrank[i]) and qrank[i] >= thr
            if in_pos:
                bars_held += 1
                hit_target = close[i] >= entry_price * (1.0 + pt)
                hit_time = bars_held >= time_stop
                if hit_target or hit_time:
                    in_pos = False
                    bars_held = 0
                    signal[i] = 0
                else:
                    signal[i] = 1
            else:
                if trigger:
                    in_pos = True
                    entry_price = close[i]
                    bars_held = 0
                    signal[i] = 1

        df = pd.DataFrame(index=data.index)
        df["signal"] = signal
        df["size"] = 1.0
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")
