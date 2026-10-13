from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class SeasonalStreakParams:
    min_streak: int = 3
    hold_bars: int = 8
    anchor_weekday: int = 0


class GeneratedStrategy(BaseStrategy[SeasonalStreakParams]):
    strategy_id = "gen_local_1790857879"

    @classmethod
    def params_type(cls):
        return SeasonalStreakParams

    def warmup_bars(self, params: SeasonalStreakParams) -> int:
        if params.min_streak < 1 or not 5 <= params.hold_bars <= 10:
            raise ValueError("min_streak must be positive; hold_bars must be 5 through 10")
        if not 0 <= params.anchor_weekday <= 4:
            raise ValueError("anchor_weekday must be 0 through 4")
        return int(7 * params.min_streak + 3)

    def indicators(self, data: pd.DataFrame, params: SeasonalStreakParams) -> pd.DataFrame:
        self.warmup_bars(params)
        out = pd.DataFrame(index=data.index)
        valid = np.isfinite(data["open"]) & np.isfinite(data["close"])
        anchor = pd.Series(data.index.dayofweek == params.anchor_weekday, index=data.index)
        out["anchor_down"] = anchor & valid & data["close"].lt(data["open"])
        weekly_down = out.loc[anchor, "anchor_down"]
        # Cap the count: only the most recent min_streak anchor sessions matter.
        streak = pd.Series(0, index=weekly_down.index, dtype=int)
        consecutive = pd.Series(True, index=weekly_down.index)
        for lag in range(params.min_streak):
            consecutive &= weekly_down.shift(lag, fill_value=False)
            streak += consecutive.astype(int)
        out["anchor_streak"] = streak.reindex(data.index, fill_value=0)
        out["recovery"] = valid & data["close"].gt(data["open"]) & data["close"].gt(data["close"].shift(1))
        out["confirmed"] = (
            out["anchor_streak"].shift(2).fillna(0).ge(params.min_streak)
            & out["recovery"].shift(1, fill_value=False)
            & out["recovery"]
        )
        return out

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: SeasonalStreakParams,
    ) -> SignalFrame:
        warmup = self.warmup_bars(params)
        entries = indicators["confirmed"].fillna(False).to_numpy(dtype=bool)
        signals = np.zeros(len(data), dtype=int)
        exit_bar = -1
        for i in range(len(data)):
            if exit_bar >= 0:
                if i < exit_bar:
                    signals[i] = 1
                else:
                    exit_bar = -1
                continue
            if i >= warmup and entries[i]:
                signals[i] = 1
                exit_bar = i + params.hold_bars
        df = pd.DataFrame({"signal": signals, "size": 0.95}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check() -> None:
    strategy = GeneratedStrategy()
    params = SeasonalStreakParams()
    index = pd.bdate_range("2020-01-06", periods=70)
    close = pd.Series(100.0 + np.arange(len(index)), index=index)
    open_price = close - 0.5
    open_price.loc[index.dayofweek == 0] = close + 0.5
    data = pd.DataFrame({
        "open": open_price, "high": np.maximum(open_price, close) + 1.0,
        "low": np.minimum(open_price, close) - 1.0,
        "close": close, "volume": 1000000.0,
    }, index=index)
    ctx = StrategyContext(symbol="SPY", timeframe="1d", warmup_bars=strategy.warmup_bars(params))
    derived = strategy.indicators(data, params)
    assert not derived.loc[index.dayofweek != 2, "confirmed"].any()
    assert derived["confirmed"].any()
    entries = pd.DataFrame({"confirmed": False}, index=index)
    start = strategy.warmup_bars(params)
    entries.iloc[start:start + params.hold_bars + 1, 0] = True
    result = strategy.generate_signals(data, entries, ctx, params).data
    expected = np.zeros(len(index), dtype=int)
    expected[start + 1:start + params.hold_bars + 1] = 1
    assert np.array_equal(result["signal"].to_numpy(), expected)
    changed = data.copy()
    changed.iloc[55:, changed.columns.get_loc("close")] *= 1.2
    pd.testing.assert_frame_equal(
        derived.iloc[:55], strategy.indicators(changed, params).iloc[:55]
    )
    empty = data.iloc[:0]
    assert strategy.generate_signals(empty, strategy.indicators(empty, params), ctx, params).data.empty


if __name__ == "__main__":
    _self_check()
