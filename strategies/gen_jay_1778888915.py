from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class Params:
    profit_target_pct: float = 1.5
    time_stop_bars: int = 2


class GeneratedStrategy(BaseStrategy["Params"]):
    strategy_id = "gen_jay_1778888915"

    @classmethod
    def params_type(cls):
        return Params

    @staticmethod
    def warmup_bars(params: Params) -> int:
        return 65

    @staticmethod
    def indicators(data: pd.DataFrame, params: Params) -> pd.DataFrame:
        ind = pd.DataFrame(index=data.index)
        window = 63

        rolling_high = data["close"].rolling(window).max()
        ind["drawdown"] = (
            (data["close"] - rolling_high) / rolling_high.replace(0, np.nan)
        )

        # Percentile rank within rolling window: low rank = currently deep drawdown
        ind["dd_rank"] = ind["drawdown"].rolling(window).rank(pct=True)

        daily_range = (data["high"] - data["low"]) / data["close"]
        rolling_mean_range = daily_range.rolling(window).mean().replace(0, np.nan)
        ind["range_ratio"] = daily_range / rolling_mean_range

        return ind

    @staticmethod
    def generate_signals(
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: Params,
    ) -> SignalFrame:
        n = len(data)
        close = data["close"].values
        dd_rank = indicators["dd_rank"].values
        range_ratio = indicators["range_ratio"].values

        signal_arr = np.zeros(n, dtype=int)
        size_arr = np.ones(n, dtype=float)

        # Two-threshold hysteresis on drawdown rank
        deep_thresh = 0.20     # rank below this ARMS the entry
        release_thresh = 0.40  # rank above this (while armed) FIRES entry
        expire_thresh = 0.65   # rank above this resets armed (context stale)
        range_compress = 0.90  # range_ratio must be below this

        profit_mult = 1.0 + params.profit_target_pct / 100.0

        armed = False
        in_trade = False
        entry_price = 0.0
        bars_held = 0

        for i in range(n):
            if np.isnan(dd_rank[i]) or np.isnan(range_ratio[i]):
                continue

            if in_trade:
                bars_held += 1
                if (
                    close[i] >= entry_price * profit_mult
                    or bars_held >= params.time_stop_bars
                ):
                    signal_arr[i] = 0
                    in_trade = False
                    armed = False
                else:
                    signal_arr[i] = 1
            else:
                # Hysteresis state machine
                if dd_rank[i] < deep_thresh:
                    armed = True
                elif dd_rank[i] > expire_thresh and armed:
                    # Deep-drawdown context expired before release fired
                    armed = False

                # Entry: armed + recovery past release threshold + range compressed
                if (
                    armed
                    and dd_rank[i] >= release_thresh
                    and range_ratio[i] < range_compress
                ):
                    signal_arr[i] = 1
                    in_trade = True
                    entry_price = close[i]
                    bars_held = 0
                    armed = False

        df = pd.DataFrame(index=data.index)
        df["signal"] = signal_arr
        df["size"] = size_arr

        # Mandatory 1-bar shift: decision at bar N fills at bar N+1
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = df["size"].shift(1).fillna(1.0)

        return SignalFrame(data=df, signal_column="signal", size_column="size")
