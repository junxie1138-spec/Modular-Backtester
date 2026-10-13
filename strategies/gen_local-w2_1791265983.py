from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class GapCycleParams:
    lookback: int = 20
    holding_bars: int = 4


class GeneratedStrategy(BaseStrategy[GapCycleParams]):
    strategy_id = "gen_local-w2_1791265983"

    @classmethod
    def params_type(cls) -> type[GapCycleParams]:
        return GapCycleParams

    @staticmethod
    def warmup_bars(params: GapCycleParams) -> int:
        if params.lookback < 2 or not 3 <= params.holding_bars <= 5:
            raise ValueError("lookback must be >= 2; holding_bars must be 3 through 5")
        return int(params.lookback + 3)

    @staticmethod
    def indicators(data: pd.DataFrame, params: GapCycleParams) -> pd.DataFrame:
        GeneratedStrategy.warmup_bars(params)
        previous_close = data["close"].shift(1).where(lambda x: x > 0)
        gap = (data["open"] / previous_close - 1.0).replace([np.inf, -np.inf], np.nan)
        gap_sign = np.sign(gap)
        transitions = gap_sign * gap_sign.shift(1)
        return pd.DataFrame({
            "gap": gap,
            "gap_floor": gap.abs().rolling(params.lookback, min_periods=params.lookback).median().shift(1),
            "gap_persistence": transitions.rolling(params.lookback, min_periods=params.lookback).mean().shift(1),
        }, index=data.index)

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: GapCycleParams,
    ) -> SignalFrame:
        GeneratedStrategy.warmup_bars(params)
        gap = indicators["gap"]
        floor = indicators["gap_floor"]
        persistence = indicators["gap_persistence"]
        ready = np.isfinite(gap) & np.isfinite(floor) & np.isfinite(persistence)
        eligible = ready & floor.gt(0) & gap.abs().ge(floor)
        direction = np.sign(gap.fillna(0)).astype(int)
        candidates = direction.where(persistence.le(0), -direction).where(eligible, 0).to_numpy(dtype=int)
        signals = np.zeros(len(data), dtype=int)
        position = 0
        exit_bar = -1
        for i, candidate in enumerate(candidates):
            if position != 0:
                if i == exit_bar:
                    position = 0
                else:
                    signals[i] = position
                continue
            if candidate != 0:
                position = int(candidate)
                exit_bar = i + params.holding_bars
                signals[i] = position
        df = pd.DataFrame({"signal": signals, "size": 0.95}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


if __name__ == "__main__":
    index = pd.date_range("2020-01-01", periods=12, freq="D")
    data = pd.DataFrame(index=index)
    derived = pd.DataFrame({
        "gap": np.ones(12),
        "gap_floor": np.full(12, 0.5),
        "gap_persistence": [-1.0] * 5 + [1.0] * 7,
    }, index=index)
    result = GeneratedStrategy.generate_signals(None, data, derived, None, GapCycleParams())
    assert result.data["signal"].tolist() == [0, 1, 1, 1, 1, 0, -1, -1, -1, -1, 0, -1]
    derived.loc[:, "gap_floor"] = np.nan
    result = GeneratedStrategy.generate_signals(None, data, derived, None, GapCycleParams())
    assert result.data["signal"].eq(0).all()
