from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class RangeInvasionParams:
    migration_bars: int = 3
    history_bars: int = 126
    entry_percentile: float = 0.90
    hold_bars: int = 8


class GeneratedStrategy(BaseStrategy[RangeInvasionParams]):
    strategy_id = "gen_local_1791259739"

    @classmethod
    def params_type(cls) -> type[RangeInvasionParams]:
        return RangeInvasionParams

    def warmup_bars(self, params: RangeInvasionParams) -> int:
        return int(params.migration_bars + params.history_bars + 1)

    def indicators(self, data: pd.DataFrame, params: RangeInvasionParams) -> pd.DataFrame:
        if params.migration_bars < 1 or params.history_bars < 2:
            raise ValueError("Lookback lengths must be positive; history_bars must be at least two.")
        if not 0.0 < params.entry_percentile < 1.0:
            raise ValueError("entry_percentile must be between zero and one.")
        result = pd.DataFrame(index=data.index)
        result["range"] = (data["high"] - data["low"]).where(data["high"] > data["low"])
        result["previous_low"] = data["low"].shift(1)
        result["previous_high"] = data["high"].shift(1)
        result["downward_fraction"] = (
            (result["previous_low"] - data["low"]).clip(lower=0)
            / result["range"]
        ).clip(upper=1)
        result["upward_fraction"] = (
            (data["high"] - result["previous_high"]).clip(lower=0)
            / result["range"]
        ).clip(upper=1)
        result["invasion"] = result["downward_fraction"] - result["upward_fraction"]
        result["migration"] = result["invasion"].rolling(
            params.migration_bars, min_periods=params.migration_bars
        ).mean()
        result["threshold"] = result["migration"].rolling(
            params.history_bars, min_periods=params.history_bars
        ).quantile(params.entry_percentile).shift(1)
        return result.replace([np.inf, -np.inf], np.nan)

    @staticmethod
    def _fixed_positions(entries: np.ndarray, hold_bars: int) -> np.ndarray:
        if hold_bars < 1:
            raise ValueError("hold_bars must be positive.")
        signal = np.zeros(len(entries), dtype=int)
        exit_bar = -1
        for i, enter in enumerate(entries):
            if exit_bar >= 0:
                if i == exit_bar:
                    exit_bar = -1
                else:
                    signal[i] = 1
            elif enter:
                exit_bar = i + hold_bars
                signal[i] = 1
        return signal

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: RangeInvasionParams,
    ) -> SignalFrame:
        entries = (
            indicators["migration"].gt(indicators["threshold"])
            & indicators["migration"].gt(0)
            & indicators["threshold"].notna()
        ).fillna(False).to_numpy(dtype=bool)
        df = pd.DataFrame(index=data.index)
        df["signal"] = self._fixed_positions(entries, params.hold_bars)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = 0.95
        return SignalFrame(data=df, signal_column="signal", size_column="size")


if __name__ == "__main__":
    decisions = GeneratedStrategy._fixed_positions(np.ones(7, dtype=bool), 2)
    assert decisions.tolist() == [1, 1, 0, 1, 1, 0, 1]
    fills = pd.Series(decisions).shift(1).fillna(0).astype(int)
    assert fills.tolist() == [0, 1, 1, 0, 1, 1, 0]
    assert GeneratedStrategy._fixed_positions(np.zeros(5, dtype=bool), 8).sum() == 0
