from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class MomentumFrontParams:
    fast_bars: int = 2
    slow_bars: int = 8
    percentile_bars: int = 126
    entry_percentile: float = 0.8
    holding_bars: int = 2


class GeneratedStrategy(BaseStrategy[MomentumFrontParams]):
    strategy_id = "gen_local-w2_1791262732"

    @classmethod
    def params_type(cls) -> type[MomentumFrontParams]:
        return MomentumFrontParams

    def warmup_bars(self, params: MomentumFrontParams) -> int:
        if not 1 <= params.fast_bars < params.slow_bars:
            raise ValueError("Require 1 <= fast_bars < slow_bars")
        if params.percentile_bars < 2 or not 0 < params.entry_percentile < 1:
            raise ValueError("Invalid percentile window or probability")
        if params.holding_bars not in (1, 2):
            raise ValueError("holding_bars must be 1 or 2")
        return int(params.slow_bars + params.percentile_bars + 2)

    def indicators(self, data: pd.DataFrame, params: MomentumFrontParams) -> pd.DataFrame:
        self.warmup_bars(params)
        out = pd.DataFrame(index=data.index)
        close = data["close"].astype(float).where(data["close"] > 0)
        out["fast_roc"] = close.div(close.shift(params.fast_bars)).sub(1)
        out["slow_roc"] = close.div(close.shift(params.slow_bars)).sub(1)
        out["fast_acceleration"] = out["fast_roc"].diff().div(params.fast_bars)
        out["slow_acceleration"] = out["slow_roc"].diff().div(params.slow_bars)
        out["acceleration_spread"] = out["fast_acceleration"] - out["slow_acceleration"]
        out = out.replace([np.inf, -np.inf], np.nan)
        out["entry_threshold"] = out["acceleration_spread"].shift(1).rolling(
            params.percentile_bars, min_periods=params.percentile_bars
        ).quantile(params.entry_percentile)
        return out

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: MomentumFrontParams,
    ) -> SignalFrame:
        self.warmup_bars(params)
        ready = indicators.notna().all(axis=1)
        entries = (
            ready
            & (indicators["slow_roc"] > 0)
            & (indicators["fast_roc"] > 0)
            & (indicators["fast_acceleration"] > 0)
            & (indicators["slow_acceleration"] <= 0)
            & (indicators["acceleration_spread"] > indicators["entry_threshold"])
        ).fillna(False).to_numpy(dtype=bool)
        decisions = self._fixed_hold(entries, params.holding_bars)
        df = pd.DataFrame(index=data.index)
        df["signal"] = decisions
        df["size"] = 0.95
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")

    @staticmethod
    def _fixed_hold(entries: np.ndarray, holding_bars: int) -> np.ndarray:
        signals = np.zeros(len(entries), dtype=int)
        exit_bar = -1
        for i, entry in enumerate(entries):
            if exit_bar >= 0:
                if i == exit_bar:
                    exit_bar = -1
                    continue
                signals[i] = 1
            elif entry:
                signals[i] = 1
                exit_bar = i + holding_bars
        return signals


if __name__ == "__main__":
    entries = np.ones(7, dtype=bool)
    assert GeneratedStrategy._fixed_hold(entries, 2).tolist() == [1, 1, 0, 1, 1, 0, 1]
    assert GeneratedStrategy._fixed_hold(entries, 1).tolist() == [1, 0, 1, 0, 1, 0, 1]
    assert GeneratedStrategy._fixed_hold(np.zeros(4, dtype=bool), 2).tolist() == [0, 0, 0, 0]
