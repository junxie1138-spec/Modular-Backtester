from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class GapParams:
    baseline_bars: int = 20
    min_gap: float = 0.001
    retained_fraction: float = 0.5
    max_prevalence: float = 0.25
    hold_bars: int = 8


class GeneratedStrategy(BaseStrategy[GapParams]):
    strategy_id = "gen_local_1790917073"

    @classmethod
    def params_type(cls) -> type[GapParams]:
        return GapParams

    def warmup_bars(self, params: GapParams) -> int:
        return max(1, int(params.baseline_bars)) + 3

    def indicators(self, data: pd.DataFrame, params: GapParams) -> pd.DataFrame:
        if params.baseline_bars < 1 or params.hold_bars < 1:
            raise ValueError("baseline_bars and hold_bars must be positive")
        if not np.isfinite(params.min_gap) or params.min_gap <= 0:
            raise ValueError("min_gap must be finite and positive")
        if not np.isfinite(params.retained_fraction) or not 0 <= params.retained_fraction <= 1:
            raise ValueError("retained_fraction must be between zero and one")
        if not np.isfinite(params.max_prevalence) or not 0 <= params.max_prevalence <= 1:
            raise ValueError("max_prevalence must be between zero and one")
        out = pd.DataFrame(index=data.index)
        out["prior_close"] = data["close"].shift(1)
        out["gap"] = (
            data["open"] / out["prior_close"].replace(0.0, np.nan) - 1.0
        ).replace([np.inf, -np.inf], np.nan).fillna(0.0)
        out["retained_fraction"] = (
            (data["low"] - out["prior_close"])
            / (data["open"] - out["prior_close"]).replace(0.0, np.nan)
        ).replace([np.inf, -np.inf], np.nan).fillna(0.0)
        out["defended"] = (
            (out["prior_close"] > 0)
            & (out["gap"] >= params.min_gap)
            & (out["retained_fraction"] >= params.retained_fraction)
            & (data["close"] >= data["open"])
        ).fillna(False)
        out["prevalence"] = (
            out["defended"].astype(float)
            .rolling(params.baseline_bars, min_periods=params.baseline_bars)
            .mean().shift(2)
        )
        out["entry"] = (
            out["defended"]
            & out["defended"].shift(1, fill_value=False)
            & out["prevalence"].notna()
            & out["prevalence"].le(params.max_prevalence)
        ).fillna(False)
        return out

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: GapParams,
    ) -> SignalFrame:
        if params.hold_bars < 1:
            raise ValueError("hold_bars must be positive")
        entry = indicators["entry"].fillna(False).to_numpy(dtype=bool)
        signal = np.zeros(len(data), dtype=int)
        exit_bar = -1
        for i in range(len(data)):
            if exit_bar >= 0:
                if i == exit_bar:
                    exit_bar = -1
                else:
                    signal[i] = 1
            elif entry[i]:
                signal[i] = 1
                exit_bar = i + params.hold_bars
        df = pd.DataFrame(index=data.index)
        df["signal"] = signal
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = 0.95
        return SignalFrame(data=df, signal_column="signal", size_column="size")


if __name__ == "__main__":
    index = pd.date_range("2020-01-01", periods=14, freq="B")
    data = pd.DataFrame(100.0, index=index, columns=["open", "high", "low", "close", "volume"])
    indicators = pd.DataFrame({"entry": [False] + [True] * 13}, index=index)
    result = GeneratedStrategy().generate_signals(data, indicators, None, GapParams(hold_bars=5)).data
    assert result["signal"].tolist() == [0, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0]
    assert result["size"].gt(0).all()
