from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class GeneratedParams:
    gap_threshold: float = 0.002
    stretch_ratio: float = 1.25
    size: float = 1.0


class GeneratedStrategy(BaseStrategy[GeneratedParams]):
    strategy_id = "gen_local_1790714397"

    @classmethod
    def params_type(cls):
        return GeneratedParams

    @classmethod
    def warmup_bars(cls, params: GeneratedParams) -> int:
        return 3

    @classmethod
    def indicators(cls, data: pd.DataFrame, params: GeneratedParams) -> pd.DataFrame:
        prev_close = data["close"].shift(1)
        gap = data["open"].div(prev_close).sub(1.0)
        prior_gap = gap.shift(1)
        same_direction = gap.mul(prior_gap).gt(0.0)
        extending = gap.abs().ge(prior_gap.abs().mul(params.stretch_ratio))
        confirmed = (
            gap.abs().ge(params.gap_threshold)
            & prior_gap.abs().ge(params.gap_threshold)
            & same_direction
            & extending
        )
        return pd.DataFrame(
            {"gap": gap, "prior_gap": prior_gap, "setup": confirmed.fillna(False)},
            index=data.index,
        )

    @classmethod
    def generate_signals(
        cls,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: GeneratedParams,
    ) -> SignalFrame:
        setup = indicators["setup"].fillna(False).astype(bool)
        direction = -np.sign(indicators["gap"].fillna(0.0)).astype(int)
        raw = pd.Series(0, index=data.index, dtype=int)
        active = 0
        for i in range(len(data)):
            if active:
                # Exit only when the two-gap entry condition reverses.
                reversed_setup = bool(setup.iloc[i]) and direction.iloc[i] == -active
                if reversed_setup or not bool(setup.iloc[i]):
                    active = 0
            if not active and bool(setup.iloc[i]):
                active = int(direction.iloc[i])
            raw.iloc[i] = active
        df = pd.DataFrame(index=data.index)
        df["signal"] = raw.shift(1).fillna(0).astype(int)
        df["size"] = float(params.size)
        return SignalFrame(data=df, signal_column="signal", size_column="size")
