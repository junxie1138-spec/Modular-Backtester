from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class WickRankParams:
    smooth_window: int = 3
    rank_window: int = 12
    entry_pct: float = 0.8


class GeneratedStrategy(BaseStrategy[WickRankParams]):
    strategy_id = "gen_local-w2_1791295613"

    @classmethod
    def params_type(cls):
        return WickRankParams

    def warmup_bars(self, params: WickRankParams) -> int:
        return params.smooth_window + params.rank_window - 1

    def indicators(self, data: pd.DataFrame, params: WickRankParams) -> pd.DataFrame:
        upper = data["high"] - np.maximum(data["open"], data["close"])
        lower = np.minimum(data["open"], data["close"]) - data["low"]
        spread = (data["high"] - data["low"]).replace(0, np.nan)
        wick = ((lower - upper) / spread).replace([np.inf, -np.inf], np.nan).fillna(0.0).clip(-1.0, 1.0)
        smooth = wick.rolling(params.smooth_window, min_periods=params.smooth_window).mean()
        rank = smooth.rolling(params.rank_window, min_periods=params.rank_window).rank(pct=True)
        return pd.DataFrame({"wick_rank": rank.fillna(0.5)}, index=data.index)

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: WickRankParams,
    ) -> SignalFrame:
        rank = indicators["wick_rank"]
        long_entry = rank >= params.entry_pct
        short_entry = rank <= 1.0 - params.entry_pct
        signals = []
        position = 0
        for go_long, go_short in zip(long_entry, short_entry):
            if position == 0:
                position = 1 if go_long else -1 if go_short else 0
            elif position == 1 and go_short:
                position = 0
            elif position == -1 and go_long:
                position = 0
            signals.append(position)
        df = pd.DataFrame(index=data.index)
        df["signal"] = signals
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = (0.25 + 1.5 * (rank - 0.5).abs()).shift(1).fillna(0.25).astype(float)
        return SignalFrame(data=df, signal_column="signal", size_column="size")
