from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class FractionalQueueParams:
    fractional_order: float = 0.4
    fractional_window: int = 32
    baseline_window: int = 63
    queue_window: int = 3
    percentile_window: int = 126
    entry_quantile: float = 0.9
    profit_target: float = 0.015
    max_hold: int = 4

    def __post_init__(self):
        windows = (self.fractional_window, self.baseline_window,
                   self.queue_window, self.percentile_window, self.max_hold)
        if any(not isinstance(x, int) or isinstance(x, bool) or x < 1 for x in windows):
            raise ValueError("Window lengths and max_hold must be positive integers.")
        if not 0 < self.fractional_order < 1 or not 0 < self.entry_quantile < 1:
            raise ValueError("Fractional order and quantile must lie between zero and one.")
        if not np.isfinite(self.profit_target) or self.profit_target <= 0:
            raise ValueError("profit_target must be finite and positive.")


class GeneratedStrategy(BaseStrategy[FractionalQueueParams]):
    strategy_id = "gen_local_1791295273"

    @classmethod
    def params_type(cls):
        return FractionalQueueParams

    def warmup_bars(self, params: FractionalQueueParams) -> int:
        return int(params.fractional_window + params.baseline_window
                   + params.queue_window + params.percentile_window + 2)

    def indicators(self, data: pd.DataFrame, params: FractionalQueueParams) -> pd.DataFrame:
        out = pd.DataFrame(index=data.index)
        close = data["close"].astype(float)
        close = close.where(np.isfinite(close) & close.gt(0))
        weights = np.ones(params.fractional_window, dtype=float)
        for k in range(1, len(weights)):
            weights[k] = -weights[k - 1] * (params.fractional_order - k + 1) / k
        # ponytail: finite fractional memory; increase fractional_window if truncation dominates.
        fractional = close.copy()
        for k in range(1, len(weights)):
            fractional = fractional + weights[k] * close.shift(k)
        out["fractional_close"] = fractional
        out["fractional_level"] = fractional / close
        out["fractional_delta"] = out["fractional_level"].diff()
        out["baseline"] = out["fractional_level"].shift(1).rolling(
            params.baseline_window, min_periods=params.baseline_window
        ).median()
        out["surplus"] = (out["fractional_level"] - out["baseline"]).clip(lower=0)
        out["queue"] = out["surplus"].rolling(
            params.queue_window, min_periods=params.queue_window
        ).sum()
        out["capacity"] = out["queue"].shift(1).rolling(
            params.percentile_window, min_periods=params.percentile_window
        ).quantile(params.entry_quantile)
        return out.replace([np.inf, -np.inf], np.nan)

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: FractionalQueueParams,
    ) -> SignalFrame:
        overflow = indicators["queue"].gt(indicators["capacity"])
        fresh = indicators["queue"].shift(1).le(indicators["capacity"].shift(1))
        entries = (overflow & fresh & indicators["fractional_delta"].gt(0)).fillna(False)
        entry_flags = entries.to_numpy(dtype=bool)
        opens = data["open"].to_numpy(dtype=float)
        closes = data["close"].to_numpy(dtype=float)
        decisions = np.zeros(len(data), dtype=int)
        active = False
        entry_bar = -1
        entry_price = np.nan
        for i in range(len(data)):
            if active:
                if i == entry_bar:
                    entry_price = opens[i]
                    if not np.isfinite(entry_price) or entry_price <= 0:
                        active = False
                        continue
                target_hit = np.isfinite(closes[i]) and closes[i] >= entry_price * (1 + params.profit_target)
                expired = i - entry_bar + 1 >= params.max_hold
                if target_hit or expired:
                    active = False
                    continue
                decisions[i] = 1
            elif entry_flags[i] and np.isfinite(closes[i]) and closes[i] > 0:
                active = True
                entry_bar = i + 1
                entry_price = np.nan
                decisions[i] = 1
        df = pd.DataFrame({"signal": decisions, "size": 1.0}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check():
    index = pd.date_range("2020-01-01", periods=8)
    data = pd.DataFrame(
        {"open": 100.0, "high": 103.0, "low": 99.0, "close": 100.0, "volume": 1000.0},
        index=index,
    )
    indicators = pd.DataFrame(
        {"queue": [0.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
         "capacity": 1.0, "fractional_delta": 1.0},
        index=index,
    )
    strategy = GeneratedStrategy()
    params = FractionalQueueParams(max_hold=3)
    ctx = StrategyContext(symbol="SPY", timeframe="1d", warmup_bars=0)
    result = strategy.generate_signals(data, indicators, ctx, params).data
    assert result["signal"].tolist() == [0, 0, 1, 1, 1, 0, 0, 0]
    data.loc[index[3], "close"] = 102.0
    result = strategy.generate_signals(data, indicators, ctx, params).data
    assert result["signal"].tolist() == [0, 0, 1, 1, 0, 0, 0, 0]
    assert result["size"].gt(0).all()


if __name__ == "__main__":
    _self_check()
