from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class GapPhaseParams:
    phase_window: int = 20
    min_gap: float = 0.003


class GeneratedStrategy(BaseStrategy[GapPhaseParams]):
    strategy_id = "gen_local-w2_1791297805"

    @classmethod
    def params_type(cls) -> type[GapPhaseParams]:
        return GapPhaseParams

    def warmup_bars(self, params: GapPhaseParams) -> int:
        return params.phase_window + 2

    def indicators(self, data: pd.DataFrame, params: GapPhaseParams) -> pd.DataFrame:
        if params.phase_window < 2 or params.min_gap <= 0:
            raise ValueError("phase_window must be at least 2 and min_gap positive")
        gap = (data["open"] / data["close"].shift(1) - 1).replace([np.inf, -np.inf], np.nan).fillna(0.0)
        agreement = np.sign(gap) * np.sign(gap.shift(1))
        phase = agreement.rolling(params.phase_window, min_periods=params.phase_window).mean().shift(1).fillna(0.0)
        return pd.DataFrame({"gap": gap, "phase": phase}, index=data.index)

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame,
                         ctx: StrategyContext, params: GapPhaseParams) -> SignalFrame:
        eligible = (indicators["gap"].abs() >= params.min_gap) & (indicators["phase"].abs() >= 0.2)
        candidate = np.where(eligible, np.sign(indicators["gap"]) * np.sign(indicators["phase"]), 0).astype(int)
        raw = np.zeros(len(data), dtype=int)
        side = 0
        exit_bar = -1
        for i, direction in enumerate(candidate):
            if side and i >= exit_bar:
                side = 0
            elif not side and direction:
                side = int(direction)
                exit_bar = i + 2
            raw[i] = side
        df = pd.DataFrame({"signal": raw, "size": 1.0}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check() -> None:
    index = pd.date_range("2020-01-01", periods=6)
    data = pd.DataFrame(index=index)
    indicators = pd.DataFrame({"gap": [0, 0.01, 0, 0, 0, 0],
                               "phase": [0, 0.5, 0.5, 0.5, 0.5, 0.5]}, index=index)
    result = GeneratedStrategy().generate_signals(data, indicators, None, GapPhaseParams())
    assert result.data["signal"].tolist() == [0, 0, 1, 1, 0, 0]


if __name__ == "__main__":
    _self_check()
