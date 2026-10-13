from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class RangeDecayParams:
    lookback: int = 126
    spike_percentile: float = 0.88
    quiet_percentile: float = 0.35
    hold_bars: int = 4


class GeneratedStrategy(BaseStrategy[RangeDecayParams]):
    strategy_id = "gen_local-w2_1791226989"

    @classmethod
    def params_type(cls) -> type[RangeDecayParams]:
        return RangeDecayParams

    @classmethod
    def warmup_bars(cls, params: RangeDecayParams) -> int:
        return params.lookback + 3

    def indicators(self, data: pd.DataFrame, params: RangeDecayParams) -> pd.DataFrame:
        prior_close = data["close"].shift(1).replace(0, np.nan)
        daily_range = ((data["high"] - data["low"]) / prior_close).replace([np.inf, -np.inf], np.nan)
        history = daily_range.rolling(params.lookback, min_periods=params.lookback)
        return pd.DataFrame({
            "daily_range": daily_range.fillna(0.0),
            "spike_cutoff": history.quantile(params.spike_percentile).shift(1).fillna(np.inf),
            "quiet_cutoff": history.quantile(params.quiet_percentile).shift(1).fillna(-np.inf),
            "shock_midpoint": ((data["high"] + data["low"]) / 2).shift(2).fillna(np.inf),
        }, index=data.index)

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: RangeDecayParams,
    ) -> SignalFrame:
        r = indicators["daily_range"]
        entry = (
            (r.shift(2) > indicators["spike_cutoff"].shift(2))
            & (r.shift(1) < r.shift(2))
            & (r < r.shift(1))
            & (r < indicators["quiet_cutoff"])
            & (data["close"] > indicators["shock_midpoint"])
        ).fillna(False)

        signal = pd.Series(0, index=data.index, dtype=int)
        exit_at = -1
        for i in range(len(data)):
            if i < exit_at:
                signal.iat[i] = 1
            elif i > exit_at and bool(entry.iat[i]):
                signal.iat[i] = 1
                exit_at = i + params.hold_bars

        df = pd.DataFrame({"signal": signal, "size": 1.0}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")
