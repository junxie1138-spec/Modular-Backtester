from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class SeasonalPredatorPreyParams:
    cycle_length: int = 21
    rank_window: int = 252
    volume_smooth: int = 5
    entry_edge: float = 0.45
    exit_edge: float = -0.05
    min_size: float = 0.25
    max_size: float = 0.95


class GeneratedStrategy(BaseStrategy[SeasonalPredatorPreyParams]):
    strategy_id = "gen_local_1791223189"

    @classmethod
    def params_type(cls) -> type[SeasonalPredatorPreyParams]:
        return SeasonalPredatorPreyParams

    @classmethod
    def warmup_bars(cls, params: SeasonalPredatorPreyParams) -> int:
        return int(params.rank_window + 2 * params.cycle_length + params.volume_smooth)

    def indicators(
        self, data: pd.DataFrame, params: SeasonalPredatorPreyParams
    ) -> pd.DataFrame:
        close = data["close"].astype(float)
        volume = data["volume"].astype(float)

        prior_cycle_return = close.pct_change(
            params.cycle_length, fill_method=None
        ).shift(params.cycle_length)
        smooth_volume = volume.rolling(
            params.volume_smooth, min_periods=params.volume_smooth
        ).mean()
        current_volume_change = smooth_volume.pct_change(
            params.cycle_length, fill_method=None
        )

        price_rank = prior_cycle_return.rolling(
            params.rank_window, min_periods=params.rank_window
        ).rank(pct=True)
        volume_rank = current_volume_change.rolling(
            params.rank_window, min_periods=params.rank_window
        ).rank(pct=True)

        result = pd.DataFrame(index=data.index)
        result["price_echo_rank"] = price_rank
        result["turnover_pressure_rank"] = volume_rank
        result["phase_edge"] = price_rank - volume_rank
        return result.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: SeasonalPredatorPreyParams,
    ) -> SignalFrame:
        edge = indicators["phase_edge"].fillna(0.0)
        position = np.zeros(len(data), dtype=int)
        holding = False

        for i, value in enumerate(edge.to_numpy(dtype=float)):
            if not holding and value >= params.entry_edge:
                holding = True
            elif holding and value <= params.exit_edge:
                holding = False
            position[i] = int(holding)

        strength = ((edge - params.entry_edge) / (1.0 - params.entry_edge)).clip(0.0, 1.0)
        raw_size = params.min_size + (params.max_size - params.min_size) * strength

        df = pd.DataFrame(index=data.index)
        df["signal"] = pd.Series(position, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = raw_size.shift(1).fillna(params.min_size).clip(
            params.min_size, params.max_size
        ).astype(float)
        return SignalFrame(data=df, signal_column="signal", size_column="size")
