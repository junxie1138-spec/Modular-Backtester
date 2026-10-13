from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class SeasonalParams:
    edge_window: int = 252
    rank_window: int = 126
    entry_percentile: float = 0.70


class GeneratedStrategy(BaseStrategy[SeasonalParams]):
    strategy_id = "gen_local_1791300592"

    @classmethod
    def params_type(cls):
        return SeasonalParams

    def warmup_bars(self, params: SeasonalParams) -> int:
        return params.edge_window + params.rank_window + 1

    def indicators(self, data: pd.DataFrame, params: SeasonalParams) -> pd.DataFrame:
        ret = data["close"].pct_change(fill_method=None).replace([np.inf, -np.inf], np.nan)
        phase = np.minimum((data.index.day - 1) // 7, 3)
        edge = pd.Series(np.nan, index=data.index, dtype=float)

        for bucket in range(4):
            mask = pd.Series(phase == bucket, index=data.index)
            sample = ret.where(mask, 0.0).fillna(0.0)
            signed = sample.rolling(params.edge_window, min_periods=params.edge_window).sum()
            movement = sample.abs().rolling(params.edge_window, min_periods=params.edge_window).sum()
            count = (mask & ret.notna()).astype(int).rolling(params.edge_window, min_periods=params.edge_window).sum()
            bucket_edge = (signed / movement.replace(0.0, np.nan)).where(count >= 20)
            edge = edge.where(~mask, bucket_edge)

        out = pd.DataFrame(index=data.index)
        out["seasonal_edge"] = edge.fillna(0.0)
        out["edge_percentile"] = edge.rolling(
            params.rank_window, min_periods=params.rank_window
        ).rank(pct=True).fillna(0.5)
        return out

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: SeasonalParams,
    ) -> SignalFrame:
        edge = indicators["seasonal_edge"]
        rank = indicators["edge_percentile"]
        long_entry = (edge > 0.0) & (rank >= params.entry_percentile)
        short_entry = (edge < 0.0) & (rank <= 1.0 - params.entry_percentile)

        positions = []
        position = 0
        for go_long, go_short in zip(long_entry, short_entry):
            if position == 1 and go_short:
                position = 0
            elif position == -1 and go_long:
                position = 0
            elif position == 0:
                if go_long:
                    position = 1
                elif go_short:
                    position = -1
            positions.append(position)

        df = pd.DataFrame(index=data.index)
        df["signal"] = positions
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = 1.0
        return SignalFrame(data=df, signal_column="signal", size_column="size")
