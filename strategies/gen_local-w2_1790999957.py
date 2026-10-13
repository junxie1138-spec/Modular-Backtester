from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class GapTransmissionParams:
    gap_window: int = 4
    session_window: int = 4
    min_acceleration: float = 0.00015
    min_efficiency: float = 0.25


class GeneratedStrategy(BaseStrategy[GapTransmissionParams]):
    strategy_id = "gen_local-w2_1790999957"

    @classmethod
    def params_type(cls) -> type[GapTransmissionParams]:
        return GapTransmissionParams

    def warmup_bars(self, params: GapTransmissionParams) -> int:
        gap_window = max(1, int(params.gap_window))
        session_window = max(1, int(params.session_window))
        return int(max(2 * gap_window + 1, session_window + 1))

    def indicators(self, data: pd.DataFrame, params: GapTransmissionParams) -> pd.DataFrame:
        gap_window = max(1, int(params.gap_window))
        session_window = max(1, int(params.session_window))
        result = pd.DataFrame(index=data.index)
        previous_close = data["close"].shift(1).replace(0.0, np.nan)
        opening_price = data["open"].replace(0.0, np.nan)
        result["overnight_return"] = (data["open"] / previous_close - 1.0).replace([np.inf, -np.inf], np.nan)
        result["session_return"] = (data["close"] / opening_price - 1.0).replace([np.inf, -np.inf], np.nan)
        result["gap_drift"] = result["overnight_return"].rolling(gap_window, min_periods=gap_window).mean()
        result["gap_acceleration"] = result["gap_drift"] - result["gap_drift"].shift(gap_window)
        result["session_net"] = result["session_return"].rolling(session_window, min_periods=session_window).sum()
        result["session_gross"] = result["session_return"].abs().rolling(session_window, min_periods=session_window).sum()
        result["session_efficiency"] = result["session_net"] / result["session_gross"].replace(0.0, np.nan)
        return result.replace([np.inf, -np.inf], np.nan)

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: GapTransmissionParams) -> SignalFrame:
        acceleration_agrees = indicators["gap_acceleration"].gt(params.min_acceleration)
        transmission_agrees = indicators["session_efficiency"].gt(params.min_efficiency)
        ready = indicators[["gap_acceleration", "session_efficiency"]].notna().all(axis=1)
        entry_condition = (ready & acceleration_agrees & transmission_agrees).fillna(False)
        df = pd.DataFrame(index=data.index)
        # Remain long while both entry predicates hold; their reversal exits.
        df["signal"] = entry_condition.astype(int)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = 0.95
        return SignalFrame(data=df, signal_column="signal", size_column="size")
