from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class GeneratedParams:
    roc_bars: int = 3
    corr_window: int = 20
    rank_window: int = 126
    accel_percentile: float = 0.80
    persistence_percentile: float = 0.70
    hold_bars: int = 8
    size: float = 1.0


class GeneratedStrategy(BaseStrategy[GeneratedParams]):
    strategy_id = "gen_local_1791301841"

    @classmethod
    def params_type(cls):
        return GeneratedParams

    def warmup_bars(self, params: GeneratedParams) -> int:
        return max(
            2 * params.roc_bars + params.rank_window + 1,
            params.corr_window + params.rank_window + 2,
        )

    def indicators(self, data: pd.DataFrame, params: GeneratedParams) -> pd.DataFrame:
        ret = data["close"].pct_change().replace([np.inf, -np.inf], np.nan)
        roc = data["close"].pct_change(params.roc_bars)
        acceleration = roc.diff(params.roc_bars).replace([np.inf, -np.inf], np.nan)
        persistence = ret.rolling(params.corr_window).corr(ret.shift(1))
        persistence = persistence.replace([np.inf, -np.inf], np.nan)

        out = pd.DataFrame(index=data.index)
        out["acceleration"] = acceleration
        out["persistence"] = persistence
        out["accel_cutoff"] = acceleration.shift(1).rolling(params.rank_window).quantile(params.accel_percentile)
        out["persistence_cutoff"] = persistence.shift(1).rolling(params.rank_window).quantile(params.persistence_percentile)
        return out

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: GeneratedParams,
    ) -> SignalFrame:
        trigger = (
            (indicators["acceleration"] > 0)
            & (indicators["acceleration"] > indicators["accel_cutoff"])
            & (indicators["persistence"] > 0)
            & (indicators["persistence"] > indicators["persistence_cutoff"])
        ).fillna(False).to_numpy(dtype=bool)

        target = np.zeros(len(data), dtype=int)
        entry_bar = -1
        for i, enter in enumerate(trigger):
            if entry_bar >= 0:
                if i - entry_bar >= params.hold_bars:
                    entry_bar = -1
                else:
                    target[i] = 1
            elif enter:
                entry_bar = i
                target[i] = 1

        df = pd.DataFrame(index=data.index)
        df["signal"] = target
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = float(params.size)
        return SignalFrame(data=df, signal_column="signal", size_column="size")
