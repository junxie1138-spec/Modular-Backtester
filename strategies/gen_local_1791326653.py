from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class OutbreakParams:
    outbreak_threshold: float = 0.9
    hold_bars: int = 4


class GeneratedStrategy(BaseStrategy[OutbreakParams]):
    strategy_id = "gen_local_1791326653"

    @classmethod
    def params_type(cls) -> type[OutbreakParams]:
        return OutbreakParams

    def warmup_bars(self, params: OutbreakParams) -> int:
        return 21

    def indicators(self, data: pd.DataFrame, params: OutbreakParams) -> pd.DataFrame:
        if not np.isfinite(params.outbreak_threshold) or params.outbreak_threshold <= 0:
            raise ValueError("outbreak_threshold must be finite and positive")
        if params.hold_bars not in (3, 4, 5):
            raise ValueError("hold_bars must be 3, 4, or 5")
        out = pd.DataFrame(index=data.index)
        out["return"] = data["close"].pct_change(fill_method=None).replace([np.inf, -np.inf], np.nan)
        out["positive_day"] = out["return"].gt(0).astype(float).where(out["return"].notna())
        out["recent_infections"] = out["positive_day"].rolling(3, min_periods=3).sum()
        out["previous_infections"] = out["recent_infections"].shift(3)
        out["prevalence"] = out["positive_day"].rolling(20, min_periods=20).mean()
        out["susceptible"] = 1.0 - out["prevalence"]
        out["reproduction"] = (out["recent_infections"] + 0.5) / (out["previous_infections"] + 0.5)
        out["outbreak_score"] = out["reproduction"] * out["susceptible"]
        out["recent_return"] = out["return"].rolling(3, min_periods=3).sum()
        out["entry"] = (
            out["outbreak_score"].gt(params.outbreak_threshold)
            & out["recent_infections"].ge(2)
            & out["recent_return"].gt(0)
            & out["return"].gt(0)
        ).fillna(False)
        return out

    @staticmethod
    def _fixed_hold(entries: np.ndarray, hold_bars: int) -> np.ndarray:
        positions = np.zeros(len(entries), dtype=int)
        exit_at = -1
        for i, enter in enumerate(entries):
            if i < exit_at:
                positions[i] = 1
            elif i == exit_at:
                # Reserve this bar for the fixed exit, even if entry remains true.
                continue
            elif enter:
                exit_at = i + hold_bars
                positions[i] = 1
        return positions

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: OutbreakParams,
    ) -> SignalFrame:
        if params.hold_bars not in (3, 4, 5):
            raise ValueError("hold_bars must be 3, 4, or 5")
        df = pd.DataFrame(index=data.index)
        entries = indicators["entry"].fillna(False).to_numpy(dtype=bool)
        df["signal"] = self._fixed_hold(entries, params.hold_bars)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = 0.95
        return SignalFrame(data=df, signal_column="signal", size_column="size")


if __name__ == "__main__":
    for holding in (3, 4, 5):
        entries = np.zeros(holding + 3, dtype=bool)
        entries[0] = True
        expected = np.array([1] * holding + [0] * 3)
        assert np.array_equal(GeneratedStrategy._fixed_hold(entries, holding), expected)
        entries[:] = True
        expected = np.array([1] * holding + [0, 1, 1])
        assert np.array_equal(GeneratedStrategy._fixed_hold(entries, holding), expected)
