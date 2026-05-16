from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class Params:
    profit_target_pct: float = 1.5
    max_hold_bars: int = 2


class GeneratedStrategy(BaseStrategy[Params]):
    strategy_id = "gen_jay_1778909155"

    @classmethod
    def params_type(cls):
        return Params

    @staticmethod
    def warmup_bars(params: Params) -> int:
        # 20-bar capacity baseline + 10-bar deficit sum + 60-bar percentile rank = 88 bars minimum
        return 92

    @staticmethod
    def indicators(data: pd.DataFrame, params: Params) -> pd.DataFrame:
        ind = pd.DataFrame(index=data.index)

        close_safe = data["close"].replace(0.0, np.nan)
        daily_range = (data["high"] - data["low"]) / close_safe

        # Rolling median range as capacity baseline
        capacity = daily_range.rolling(20, min_periods=20).median()

        # Positive when bar range is BELOW capacity (queue filling)
        range_deficit = capacity - daily_range

        # Accumulated saturation: large positive = many consecutive sub-capacity bars
        saturation_raw = range_deficit.rolling(10, min_periods=10).sum()

        # Percentile rank within a 60-bar window
        saturation_pct = saturation_raw.rolling(60, min_periods=60).rank(pct=True)

        # Close position within recent 10-bar price range
        period_low = data["low"].rolling(10, min_periods=10).min()
        period_high = data["high"].rolling(10, min_periods=10).max()
        range_span = (period_high - period_low).replace(0.0, np.nan)
        position_in_range = (data["close"] - period_low) / range_span

        ind["saturation_pct"] = saturation_pct
        ind["position_in_range"] = position_in_range
        return ind

    @staticmethod
    def generate_signals(
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: Params,
    ) -> SignalFrame:
        close = data["close"].values
        sat_pct = indicators["saturation_pct"].values
        pos_in_range = indicators["position_in_range"].values

        n = len(close)
        signals = np.zeros(n, dtype=int)
        profit_target = params.profit_target_pct / 100.0
        max_hold = int(params.max_hold_bars)

        in_trade = False
        entry_price = 0.0
        bars_held = 0

        for i in range(n):
            if in_trade:
                bars_held += 1
                gain = (close[i] - entry_price) / entry_price if entry_price > 0.0 else 0.0
                if gain >= profit_target or bars_held >= max_hold:
                    signals[i] = 0
                    in_trade = False
                    bars_held = 0
                else:
                    signals[i] = 1
            else:
                # numpy NaN comparisons always return False, so NaN inputs safely skip entry
                if sat_pct[i] > 0.70 and pos_in_range[i] < 0.25:
                    signals[i] = 1
                    in_trade = True
                    entry_price = close[i]
                    bars_held = 0

        df = pd.DataFrame(index=data.index)
        df["signal"] = signals
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = 1.0
        return SignalFrame(data=df, signal_column="signal", size_column="size")
