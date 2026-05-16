from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class Params:
    range_ma: int = 20
    accel_smooth: int = 5
    intraday_min: float = 0.60
    hold_bars: int = 8
    size_min: float = 0.30


class GeneratedStrategy(BaseStrategy[Params]):
    strategy_id = "gen_jay_1778916399"

    @classmethod
    def params_type(cls) -> type[Params]:
        return Params

    @staticmethod
    def warmup_bars(params: Params) -> int:
        rank_window = max(params.range_ma, 30)
        return params.range_ma + params.accel_smooth + rank_window + 5

    @staticmethod
    def indicators(data: pd.DataFrame, params: Params) -> pd.DataFrame:
        ind = pd.DataFrame(index=data.index)

        raw_range = data["high"] - data["low"]
        range_ma = raw_range.rolling(params.range_ma).mean()
        # Normalized range: dimensionless, comparable across regimes
        norm_range = raw_range / range_ma

        # First derivative of normalized range (velocity)
        range_vel = norm_range.diff(1)
        # Second derivative (acceleration), smoothed to reduce noise
        range_accel = range_vel.diff(1)
        ind["accel_smooth"] = range_accel.rolling(params.accel_smooth).mean()

        # Is today's range above its own MA? (expanding regime)
        ind["is_expanding"] = (norm_range > 1.0).astype(float)

        # Intraday close position: where close falls within high-low range
        hl = raw_range.replace(0, np.nan)
        ind["intraday_pos"] = ((data["close"] - data["low"]) / hl).fillna(0.5)

        # Rolling percentile rank of smoothed acceleration for signal-scaled sizing
        rank_window = max(params.range_ma, 30)
        ind["accel_rank"] = (
            ind["accel_smooth"].rolling(rank_window).rank(pct=True).fillna(0.5)
        )

        return ind

    @staticmethod
    def generate_signals(
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: Params,
    ) -> SignalFrame:
        df = pd.DataFrame(index=data.index)

        accel = indicators["accel_smooth"].values
        is_exp = indicators["is_expanding"].values
        intraday = indicators["intraday_pos"].values
        accel_rank = indicators["accel_rank"].fillna(0.5).values

        n = len(df)
        signals = np.zeros(n, dtype=int)
        sizes = np.full(n, 0.5)

        in_position = False
        exit_bar = -1
        carry_size = 0.5

        for i in range(n):
            # Fixed-bar exit: close position exactly hold_bars after entry
            if in_position and i >= exit_bar:
                in_position = False

            if in_position:
                signals[i] = 1
                sizes[i] = carry_size
            elif (
                np.isfinite(accel[i])
                and accel[i] > 0.0
                and is_exp[i] > 0.5
                and np.isfinite(intraday[i])
                and intraday[i] > params.intraday_min
            ):
                in_position = True
                exit_bar = i + params.hold_bars
                # Signal-scaled size: larger position when acceleration rank is higher
                rank = accel_rank[i] if np.isfinite(accel_rank[i]) else 0.5
                sz = params.size_min + (1.0 - params.size_min) * rank
                carry_size = float(np.clip(sz, params.size_min, 1.0))
                signals[i] = 1
                sizes[i] = carry_size

        df["signal"] = signals
        df["size"] = sizes

        # Mandatory one-bar shift: decision made at bar N close, fill at bar N+1 open
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = df["size"].shift(1).fillna(1.0)
        df["size"] = df["size"].clip(0.1, 1.0)

        return SignalFrame(data=df, signal_column="signal", size_column="size")
