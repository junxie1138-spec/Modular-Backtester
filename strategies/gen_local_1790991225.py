from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class RangePackingParams:
    coil_bars: int = 4
    history_bars: int = 63
    packing_quantile: float = 0.35
    range_quantile: float = 0.60


class GeneratedStrategy(BaseStrategy[RangePackingParams]):
    strategy_id = "gen_local_1790991225"

    @classmethod
    def params_type(cls) -> type[RangePackingParams]:
        return RangePackingParams

    def warmup_bars(self, params: RangePackingParams) -> int:
        return int(max(params.history_bars + params.coil_bars + 3, 2 * params.coil_bars))

    def indicators(self, data: pd.DataFrame, params: RangePackingParams) -> pd.DataFrame:
        out = pd.DataFrame(index=data.index)
        window = params.coil_bars
        history = params.history_bars
        out["daily_range"] = (data["high"] - data["low"]).clip(lower=0.0)
        out["relative_range"] = out["daily_range"] / data["close"].where(data["close"] > 0.0)
        out["envelope_high"] = data["high"].rolling(window, min_periods=window).max()
        out["envelope_low"] = data["low"].rolling(window, min_periods=window).min()
        out["envelope_span"] = out["envelope_high"] - out["envelope_low"]
        out["range_mass"] = out["daily_range"].rolling(window, min_periods=window).sum()
        out["packing"] = out["envelope_span"] / out["range_mass"].replace(0.0, np.nan)
        out["smooth_packing"] = out["packing"].rolling(3, min_periods=3).mean()
        out["packing_threshold"] = out["smooth_packing"].shift(1).rolling(history, min_periods=history).quantile(params.packing_quantile)
        out["median_range"] = out["relative_range"].rolling(window, min_periods=window).median()
        out["range_threshold"] = out["relative_range"].shift(1).rolling(history, min_periods=history).quantile(params.range_quantile)
        out["previous_floor"] = out["envelope_low"].shift(window)
        out["envelope_midpoint"] = (out["envelope_high"] + out["envelope_low"]) / 2.0
        return out.replace([np.inf, -np.inf], np.nan)

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: RangePackingParams) -> SignalFrame:
        required = ["smooth_packing", "packing_threshold", "median_range", "range_threshold", "envelope_low", "previous_floor", "envelope_midpoint"]
        ready = indicators[required].notna().all(axis=1) & data["close"].notna()
        entry_condition = (
            ready
            & (indicators["smooth_packing"] < indicators["packing_threshold"])
            & (indicators["median_range"] < indicators["range_threshold"])
            & (indicators["envelope_low"] > indicators["previous_floor"])
            & (data["close"] > indicators["envelope_midpoint"])
        ).fillna(False)
        df = pd.DataFrame(index=data.index)
        # Remain long exactly while the entry condition holds; its reversal exits.
        df["signal"] = entry_condition.astype(int)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = 1.0
        return SignalFrame(data=df, signal_column="signal", size_column="size")
