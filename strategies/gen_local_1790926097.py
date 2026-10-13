from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class AbsorptionParams:
    range_lookback: int = 20
    atr_lookback: int = 14
    range_expansion: float = 1.4
    range_contraction: float = 0.8
    min_gap_atr: float = 0.3
    trail_atr: float = 2.5
    max_hold_bars: int = 10


class GeneratedStrategy(BaseStrategy[AbsorptionParams]):
    strategy_id = "gen_local_1790926097"

    @classmethod
    def params_type(cls) -> type[AbsorptionParams]:
        return AbsorptionParams

    @staticmethod
    def warmup_bars(params: AbsorptionParams) -> int:
        return int(max(max(1, params.range_lookback) + 2,
                       max(1, params.atr_lookback) + 1))

    @staticmethod
    def indicators(data: pd.DataFrame, params: AbsorptionParams) -> pd.DataFrame:
        previous_close = data["close"].shift(1)
        daily_range = data["high"] - data["low"]
        true_range = pd.concat([
            daily_range,
            (data["high"] - previous_close).abs(),
            (data["low"] - previous_close).abs(),
        ], axis=1).max(axis=1)
        true_range = true_range.where(previous_close.notna())
        atr = true_range.rolling(max(1, params.atr_lookback)).mean()
        baseline = daily_range.rolling(max(1, params.range_lookback)).mean().shift(2)
        previous_range = daily_range.shift(1)
        previous_atr = atr.shift(1)
        return pd.DataFrame({
            "atr": atr,
            "range_expansion": previous_range / baseline.where(baseline > 0),
            "range_contraction": daily_range / previous_range.where(previous_range > 0),
            "gap_atr": (data["open"] - previous_close) / previous_atr.where(previous_atr > 0),
            "repair": data["close"] - previous_close,
        }, index=data.index)

    @staticmethod
    def generate_signals(
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: AbsorptionParams,
    ) -> SignalFrame:
        valid = np.isfinite(indicators).all(axis=1)
        range_agrees = (
            (indicators["range_expansion"] >= params.range_expansion)
            & (indicators["range_contraction"] <= params.range_contraction)
        )
        gap_agrees = (
            (indicators["gap_atr"] <= -max(0.0, params.min_gap_atr))
            & (indicators["repair"] >= 0.0)
        )
        entries = (valid & range_agrees & gap_agrees).to_numpy(dtype=bool)
        closes = data["close"].to_numpy(dtype=float)
        atrs = indicators["atr"].to_numpy(dtype=float)
        targets = np.zeros(len(data), dtype=int)
        active = False
        high_water = np.nan
        stop = -np.inf
        held = 0
        hold_limit = max(1, params.max_hold_bars)
        trail_multiple = max(0.01, params.trail_atr)

        for i in range(len(data)):
            if active:
                # The first active bar is the fill bar after the entry decision.
                held += 1
                close = closes[i]
                atr = atrs[i]
                if not np.isfinite(close) or not np.isfinite(atr) or atr <= 0:
                    active = False
                else:
                    high_water = close if np.isnan(high_water) else max(high_water, close)
                    stop = max(stop, high_water - trail_multiple * atr)
                    if close <= stop or held >= hold_limit:
                        active = False
                targets[i] = int(active)
            elif entries[i]:
                active = True
                high_water = np.nan
                stop = -np.inf
                held = 0
                targets[i] = 1

        df = pd.DataFrame({"signal": targets, "size": 1.0}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")
