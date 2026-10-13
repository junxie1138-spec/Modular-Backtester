from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class GapPathParams:
    channel_bars: int = 20
    min_gap_bps: float = 8.0


class GeneratedStrategy(BaseStrategy[GapPathParams]):
    strategy_id = "gen_local-w2_1791300859"

    @classmethod
    def params_type(cls) -> type[GapPathParams]:
        return GapPathParams

    @classmethod
    def warmup_bars(cls, params: GapPathParams) -> int:
        return params.channel_bars + 2

    def indicators(self, data: pd.DataFrame, params: GapPathParams) -> pd.DataFrame:
        gap = data["open"].div(data["close"].shift(1)).sub(1.0)
        gap = gap.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        gap = gap.where(gap.abs().ge(params.min_gap_bps / 10000.0), 0.0)
        path = gap.cumsum()
        prior = path.shift(1)
        high = prior.rolling(params.channel_bars, min_periods=params.channel_bars).max().fillna(path)
        low = prior.rolling(params.channel_bars, min_periods=params.channel_bars).min().fillna(path)
        return pd.DataFrame({"gap_path": path, "prior_high": high, "prior_low": low}, index=data.index)

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: GapPathParams,
    ) -> SignalFrame:
        up = indicators["gap_path"].gt(indicators["prior_high"])
        down = indicators["gap_path"].lt(indicators["prior_low"])
        entries = up & up.shift(1, fill_value=False)
        exits = down & down.shift(1, fill_value=False)
        position = 0
        signals = []
        for enter, exit_now in zip(entries, exits):
            if position == 0 and enter:
                position = 1
            elif position == 1 and exit_now:
                position = 0
            signals.append(position)
        df = pd.DataFrame(index=data.index)
        df["signal"] = pd.Series(signals, index=data.index, dtype=int)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = 1.0
        return SignalFrame(data=df, signal_column="signal", size_column="size")
