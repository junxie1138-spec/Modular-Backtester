from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


# Fixed structural constants - kept off the tunable surface so the strategy
# stays at <=2 params (the hard twist).
_TIME_STOP_BARS = 2      # max bars to hold a position; matches the 1-2 day horizon
_RANK_WINDOW = 100       # lookback for ranking the efficiency ratio against itself
_NODE_PCT = 0.33         # efficiency must sit in its bottom third to count as a 'node'


@dataclass(slots=True)
class StandingWaveParams:
    er_window: int = 20
    profit_target: float = 0.02


class GeneratedStrategy(BaseStrategy[StandingWaveParams]):
    strategy_id = "gen_a1_1778884103"

    @classmethod
    def params_type(cls):
        return StandingWaveParams

    @staticmethod
    def warmup_bars(params: StandingWaveParams) -> int:
        return int(params.er_window) + _RANK_WINDOW + 5

    @staticmethod
    def indicators(data: pd.DataFrame, params: StandingWaveParams) -> pd.DataFrame:
        close = data["close"].astype(float)
        ret = close.pct_change()

        n = max(int(params.er_window), 2)
        # Path efficiency of close-to-close returns: net displacement over the
        # window divided by the total distance travelled. Low values mean the
        # price oscillated in place - a standing wave / node regime.
        net = ret.rolling(n, min_periods=n).sum().abs()
        path = ret.abs().rolling(n, min_periods=n).sum()
        path = path.where(path > 0.0)
        er = (net / path).clip(0.0, 1.0)

        # Relative position of today's efficiency within its own recent
        # history - a parameter-free percentile rank.
        er_pct = er.rolling(_RANK_WINDOW, min_periods=20).rank(pct=True)

        node = er_pct <= _NODE_PCT          # low-efficiency standing-wave regime
        trough = ret < 0.0                  # price on the down-swing of the wave
        entry = (node & trough).fillna(False).astype(int)

        out = pd.DataFrame(index=data.index)
        out["ret"] = ret
        out["er"] = er
        out["er_pct"] = er_pct
        out["entry"] = entry
        return out

    @staticmethod
    def generate_signals(data: pd.DataFrame, indicators: pd.DataFrame,
                         ctx: StrategyContext, params: StandingWaveParams) -> SignalFrame:
        close = data["close"].astype(float).to_numpy()
        entry = indicators["entry"].fillna(0).astype(int).to_numpy()
        n = len(close)

        target = float(params.profit_target)
        raw = np.zeros(n, dtype=np.int64)

        in_pos = False
        entry_price = 0.0
        bars_held = 0

        # Path-dependent exit: long until +profit_target gain OR the time-stop,
        # whichever fires first. A bar-indexed loop is the clean form here.
        for i in range(n):
            if in_pos:
                bars_held += 1
                gain = (close[i] / entry_price - 1.0) if entry_price > 0.0 else 0.0
                if gain >= target or bars_held >= _TIME_STOP_BARS:
                    in_pos = False
                    raw[i] = 0
                else:
                    raw[i] = 1
            else:
                if entry[i] == 1:
                    in_pos = True
                    entry_price = close[i]
                    bars_held = 0
                    raw[i] = 1
                else:
                    raw[i] = 0

        df = pd.DataFrame(index=data.index)
        df["signal"] = raw
        # Mandatory one-bar shift: decide on bar N close, fill on bar N+1.
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = 1.0
        return SignalFrame(data=df, signal_column="signal", size_column="size")
