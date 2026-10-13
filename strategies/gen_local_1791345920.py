from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class SeasonalShockParams:
    lookback: int = 6
    shock_multiple: float = 1.5
    holding_bars: int = 4


class GeneratedStrategy(BaseStrategy[SeasonalShockParams]):
    strategy_id = "gen_local_1791345920"

    @classmethod
    def params_type(cls) -> type[SeasonalShockParams]:
        return SeasonalShockParams

    def warmup_bars(self, params: SeasonalShockParams) -> int:
        if not 1 <= params.lookback <= 8:
            raise ValueError("lookback must be between 1 and 8")
        if not 3 <= params.holding_bars <= 5:
            raise ValueError("holding_bars must be between 3 and 5")
        if not np.isfinite(params.shock_multiple) or params.shock_multiple <= 0:
            raise ValueError("shock_multiple must be finite and positive")
        return int(params.lookback + 2)

    def indicators(self, data: pd.DataFrame, params: SeasonalShockParams) -> pd.DataFrame:
        self.warmup_bars(params)
        returns = data["close"].pct_change(fill_method=None).replace([np.inf, -np.inf], np.nan)
        baseline = returns.abs().shift(1).rolling(params.lookback, min_periods=params.lookback).mean()
        weekday = pd.Series(data.index.dayofweek, index=data.index)
        phase = pd.Series(np.where(weekday.eq(4), -1, np.where(weekday.eq(0), 1, 0)), index=data.index)
        eligible = baseline.gt(0) & returns.abs().ge(params.shock_multiple * baseline)
        entry = (np.sign(returns.fillna(0)) * phase).where(eligible, 0).astype(int)
        return pd.DataFrame({
            "return": returns.fillna(0),
            "prior_mean_absolute_return": baseline.fillna(0),
            "weekday": weekday,
            "seasonal_phase": phase,
            "entry": entry,
        }, index=data.index)

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: SeasonalShockParams) -> SignalFrame:
        self.warmup_bars(params)
        entries = indicators["entry"].reindex(data.index).fillna(0).to_numpy(dtype=int)
        targets = np.zeros(len(data), dtype=int)
        position = 0
        exit_at = -1
        for i, entry in enumerate(entries):
            if position:
                if i == exit_at:
                    position = 0
            elif entry:
                position = int(entry)
                exit_at = i + params.holding_bars
            targets[i] = position
        df = pd.DataFrame({"signal": targets, "size": 1.0}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check() -> None:
    strategy = GeneratedStrategy()
    params = SeasonalShockParams()
    index = pd.date_range("2024-01-01", periods=16, freq="B")
    data = pd.DataFrame({"close": 100.0}, index=index)
    indicators = pd.DataFrame({"entry": 1}, index=index)
    result = strategy.generate_signals(data, indicators, None, params).data
    assert result["signal"].tolist() == [0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0]
    indicators["entry"] = -1
    result = strategy.generate_signals(data, indicators, None, params).data
    assert result["signal"].tolist() == [0, -1, -1, -1, -1, 0, -1, -1, -1, -1, 0, -1, -1, -1, -1, 0]
    assert strategy.warmup_bars(SeasonalShockParams(lookback=8)) == 10


if __name__ == "__main__":
    _self_check()
