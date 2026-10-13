from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class StrategyParams:
    fractional_order: float = 0.5
    min_signal_noise: float = 0.25
    profit_target: float = 0.025
    holding_bars: int = 8

    def __post_init__(self):
        if not (0.0 < self.fractional_order < 1.0):
            raise ValueError("fractional_order must be between zero and one")
        if not (0.0 < self.min_signal_noise <= 1.0):
            raise ValueError("min_signal_noise must be in (0, 1]")
        if not (0.0 < self.profit_target < 1.0):
            raise ValueError("profit_target must be in (0, 1)")
        if not isinstance(self.holding_bars, int) or not 5 <= self.holding_bars <= 10:
            raise ValueError("holding_bars must be an integer from five to ten")


class GeneratedStrategy(BaseStrategy[StrategyParams]):
    strategy_id = "gen_local_1791315202"

    @classmethod
    def params_type(cls):
        return StrategyParams

    def warmup_bars(self, params: StrategyParams) -> int:
        return 8

    def indicators(self, data: pd.DataFrame, params: StrategyParams) -> pd.DataFrame:
        out = pd.DataFrame(index=data.index)
        close = data["close"].astype(float).where(lambda x: np.isfinite(x) & (x > 0))
        fractional_close = close.copy()
        weight = 1.0
        # ponytail: five taps approximate fractional memory; longer kernels need a larger warmup budget.
        for lag in range(1, 5):
            weight *= (lag - 1 - params.fractional_order) / lag
            fractional_close = fractional_close + weight * close.shift(lag)
        out["fractional_close"] = fractional_close
        out["fractional_change"] = fractional_close.diff()
        out["fractional_flow"] = out["fractional_change"].rolling(3, min_periods=3).sum()
        out["fractional_noise"] = out["fractional_change"].abs().rolling(3, min_periods=3).sum()
        out["signal_noise"] = (
            out["fractional_flow"] / out["fractional_noise"].replace(0.0, np.nan)
        ).replace([np.inf, -np.inf], np.nan)
        out["seasonal_window"] = (data.index.day >= 25) | (data.index.day <= 3)
        out["entry"] = (
            out["seasonal_window"]
            & out["signal_noise"].ge(params.min_signal_noise)
            & out["fractional_change"].gt(0.0)
        ).fillna(False)
        return out

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: StrategyParams,
    ) -> SignalFrame:
        df = pd.DataFrame(index=data.index)
        signals = np.zeros(len(data), dtype=int)
        entries = indicators["entry"].fillna(False).to_numpy(dtype=bool)
        opens = data["open"].to_numpy(dtype=float)
        closes = data["close"].to_numpy(dtype=float)
        entry_bar = -1
        entry_price = np.nan
        for i in range(len(data)):
            if entry_bar >= 0:
                if i == entry_bar:
                    entry_price = opens[i]
                invalid_fill = not np.isfinite(entry_price) or entry_price <= 0.0
                target_hit = (
                    np.isfinite(closes[i])
                    and closes[i] >= entry_price * (1.0 + params.profit_target)
                )
                time_up = i - entry_bar + 1 >= params.holding_bars
                if invalid_fill or target_hit or time_up:
                    entry_bar = -1
                    entry_price = np.nan
                    continue
                signals[i] = 1
            elif entries[i]:
                signals[i] = 1
                entry_bar = i + 1
        df["signal"] = signals
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = 0.95
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check():
    strategy = GeneratedStrategy()
    params = StrategyParams(holding_bars=5)
    index = pd.bdate_range("2024-01-01", periods=16)
    data = pd.DataFrame(
        {"open": 100.0, "high": 101.0, "low": 99.0, "close": 100.0, "volume": 1000.0},
        index=index,
    )
    ctx = StrategyContext(symbol="SPY", timeframe="1d", warmup_bars=8)
    forced = pd.DataFrame({"entry": [True] + [False] * 15}, index=index)
    timed = strategy.generate_signals(data, forced, ctx, params).data
    assert timed["signal"].tolist() == [0, 1, 1, 1, 1, 1] + [0] * 10
    data.loc[index[2], "close"] = 103.0
    targeted = strategy.generate_signals(data, forced, ctx, params).data
    assert targeted["signal"].tolist() == [0, 1, 1] + [0] * 13
    data.loc[index[1], "open"] = 102.0
    actual_fill = strategy.generate_signals(data, forced, ctx, params).data
    assert actual_fill["signal"].tolist() == timed["signal"].tolist()
    derived = strategy.indicators(data, params)
    prefix = strategy.indicators(data.iloc[:10], params)
    pd.testing.assert_frame_equal(derived.iloc[:10], prefix)
    assert not derived["entry"].iloc[:7].any()
    assert set(targeted["signal"].unique()) <= {0, 1}
    assert targeted["size"].gt(0).all()
    assert strategy.warmup_bars(params) <= 10


if __name__ == "__main__":
    _self_check()
