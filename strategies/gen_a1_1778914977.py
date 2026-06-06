from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class Params:
    window: int = 20
    hold_bars: int = 7


class GeneratedStrategy(BaseStrategy[Params]):
    strategy_id = "gen_a1_1778914977"

    # Fixed (non-tunable) structural constants - keeps tunable params <= 2.
    COMPRESSION_THRESHOLD = 0.70
    CAPACITY_MULT = 5
    TREND_MULT = 10

    @classmethod
    def params_type(cls) -> type[Params]:
        return Params

    @staticmethod
    def warmup_bars(params: Params) -> int:
        w = max(int(params.window), 2)
        # trend MA is the longest lookback; pad for diff/pct_change chains.
        return w * GeneratedStrategy.TREND_MULT + w + 5

    @staticmethod
    def indicators(data: pd.DataFrame, params: Params) -> pd.DataFrame:
        w = max(int(params.window), 2)
        close = data["close"]
        high = data["high"]
        low = data["low"]

        out = pd.DataFrame(index=data.index)

        # Volatility "queue": rolling occupancy of normalized bar range.
        safe_close = close.replace(0.0, np.nan)
        range_pct = (high - low) / safe_close
        occupancy = range_pct.rolling(w).sum()
        # Capacity = the recent ceiling the queue has reached.
        capacity = occupancy.rolling(w * GeneratedStrategy.CAPACITY_MULT).max()
        occ_ratio = occupancy / capacity.replace(0.0, np.nan)
        out["occ_ratio"] = occ_ratio

        # Rate-of-change acceleration: first difference of ROC.
        roc = close.pct_change(w)
        out["roc"] = roc
        out["accel"] = roc.diff()

        # Bull-regime gate.
        out["trend"] = close.rolling(w * GeneratedStrategy.TREND_MULT).mean()

        return out

    @staticmethod
    def generate_signals(
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: Params,
    ) -> SignalFrame:
        hold = max(int(params.hold_bars), 1)

        close = data["close"]
        occ_ratio = indicators["occ_ratio"]
        accel = indicators["accel"]
        trend = indicators["trend"]

        # Drained queue: occupancy well below its capacity ceiling.
        compressed = (occ_ratio < GeneratedStrategy.COMPRESSION_THRESHOLD).fillna(False)
        # Ignition: acceleration crosses up through zero.
        ignition = ((accel > 0) & (accel.shift(1) <= 0)).fillna(False)
        bull = (close > trend).fillna(False)

        entry_raw = (compressed & ignition & bull).to_numpy()

        n = len(data)
        position = np.zeros(n, dtype=np.int64)
        i = 0
        # Fixed-bar exit: hold exactly `hold` bars after entry, then flat.
        while i < n:
            if entry_raw[i]:
                end = min(i + hold, n)
                position[i:end] = 1
                i = end
            else:
                i += 1

        df = pd.DataFrame(index=data.index)
        df["signal"] = position
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = 1.0

        return SignalFrame(data=df, signal_column="signal", size_column="size")
