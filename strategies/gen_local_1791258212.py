from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class Params:
    min_down_streak: int = 3
    range_window: int = 20
    contraction_ratio: float = 0.8


class GeneratedStrategy(BaseStrategy[Params]):
    strategy_id = "gen_local_1791258212"

    @classmethod
    def params_type(cls):
        return Params

    def warmup_bars(self, params: Params) -> int:
        return max(params.range_window + 1, params.min_down_streak + 2)

    def indicators(self, data: pd.DataFrame, params: Params) -> pd.DataFrame:
        if params.min_down_streak < 1 or params.range_window < 2:
            raise ValueError("Streak must be positive and range window at least two.")
        if not 0.0 < params.contraction_ratio < 1.0:
            raise ValueError("Contraction ratio must lie between zero and one.")
        out = pd.DataFrame(index=data.index)
        out["change"] = data["close"].diff()
        down = out["change"].lt(0)
        out["down_streak"] = down.groupby((~down).cumsum()).cumsum().clip(upper=params.min_down_streak)
        out["prior_down_streak"] = out["down_streak"].shift(1).fillna(0)
        out["relative_range"] = (data["high"] - data["low"]) / data["close"].where(data["close"].gt(0))
        out["baseline_range"] = out["relative_range"].rolling(params.range_window, min_periods=params.range_window).median().shift(1)
        out["streak_break"] = out["change"].gt(0) & out["prior_down_streak"].ge(params.min_down_streak)
        out["range_exhaustion"] = out["relative_range"].gt(0) & out["baseline_range"].gt(0) & out["relative_range"].le(params.contraction_ratio * out["baseline_range"])
        return out.replace([np.inf, -np.inf], np.nan)

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: Params) -> SignalFrame:
        entry = indicators["streak_break"].fillna(False) & indicators["range_exhaustion"].fillna(False)
        df = pd.DataFrame(index=data.index)
        # Exposure reverses to flat when either entry primitive flips.
        df["signal"] = entry.astype(int)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = 1.0
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check() -> None:
    close = pd.Series([100.0] * 20 + [99.0, 98.0, 97.0, 97.5, 98.0])
    data = pd.DataFrame({"open": close, "high": close + 1.0, "low": close - 1.0, "close": close, "volume": 1000.0})
    data.index = pd.date_range("2020-01-01", periods=len(data))
    data.iloc[23, data.columns.get_loc("high")] = 97.75
    data.iloc[23, data.columns.get_loc("low")] = 97.25
    strategy, params = GeneratedStrategy(), Params()
    ctx = StrategyContext("SPY", "1d", strategy.warmup_bars(params))
    ind = strategy.indicators(data, params)
    result = strategy.generate_signals(data, ind, ctx, params).data
    assert result["signal"].tolist() == [0] * 24 + [1]
    assert result["size"].gt(0).all()
    extended = pd.concat([data, data.iloc[[-1]].set_axis([data.index[-1] + pd.Timedelta(days=1)])])
    signals = strategy.generate_signals(extended, strategy.indicators(extended, params), ctx, params).data["signal"]
    assert signals.iloc[-1] == 0
    wide = data.copy()
    wide.iloc[23, wide.columns.get_loc("high")] = 99.5
    wide.iloc[23, wide.columns.get_loc("low")] = 95.5
    assert strategy.generate_signals(wide, strategy.indicators(wide, params), ctx, params).data["signal"].eq(0).all()
    pd.testing.assert_frame_equal(ind.iloc[:24], strategy.indicators(data.iloc[:24], params))


if __name__ == "__main__":
    _self_check()
