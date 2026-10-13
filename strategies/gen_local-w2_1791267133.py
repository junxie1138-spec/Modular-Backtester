from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class GapPressureParams:
    pressure_window: int = 4
    range_window: int = 20
    threshold: float = 0.12


class GeneratedStrategy(BaseStrategy[GapPressureParams]):
    strategy_id = "gen_local-w2_1791267133"

    @classmethod
    def params_type(cls) -> type[GapPressureParams]:
        return GapPressureParams

    def warmup_bars(self, params: GapPressureParams) -> int:
        return max(1, params.range_window) + max(1, params.pressure_window) + 2

    def indicators(self, data: pd.DataFrame, params: GapPressureParams) -> pd.DataFrame:
        out = pd.DataFrame(index=data.index)
        out["previous_close"] = data["close"].shift(1)
        out["gap"] = data["open"] - out["previous_close"]
        out["settlement_change"] = data["close"] - out["previous_close"]
        out["gap_direction"] = np.sign(out["gap"])
        out["surviving_magnitude"] = (
            out["gap_direction"] * out["settlement_change"]
        ).clip(lower=0).clip(upper=out["gap"].abs())
        out["surviving_gap"] = out["gap_direction"] * out["surviving_magnitude"]
        out["true_range"] = pd.concat(
            [
                data["high"] - data["low"],
                (data["high"] - out["previous_close"]).abs(),
                (data["low"] - out["previous_close"]).abs(),
            ],
            axis=1,
        ).max(axis=1)
        range_window = max(1, params.range_window)
        pressure_window = max(1, params.pressure_window)
        out["prior_range_scale"] = out["true_range"].rolling(
            range_window, min_periods=range_window
        ).mean().shift(1).replace(0, np.nan)
        out["normalized_surviving_gap"] = (
            out["surviving_gap"] / out["prior_range_scale"]
        ).replace([np.inf, -np.inf], np.nan)
        out["pressure"] = out["normalized_surviving_gap"].rolling(
            pressure_window, min_periods=pressure_window
        ).mean()
        return out

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: GapPressureParams,
    ) -> SignalFrame:
        threshold = max(abs(float(params.threshold)), 1e-12)
        pressure = indicators["pressure"]
        transitions = pd.Series(np.nan, index=data.index, dtype=float)
        transitions.loc[pressure.ge(threshold)] = 1.0
        transitions.loc[pressure.le(-threshold)] = 0.0
        df = pd.DataFrame(index=data.index)
        df["signal"] = transitions.ffill().fillna(0).astype(int)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = 0.95
        return SignalFrame(data=df, signal_column="signal", size_column="size")
