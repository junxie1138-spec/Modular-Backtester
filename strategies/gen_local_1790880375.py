from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class GapMemoryParams:
    fractional_order: float = 0.45
    impulse_threshold: float = 0.06
    hold_bars: int = 4

    def __post_init__(self):
        if not 0.0 < self.fractional_order < 1.0:
            raise ValueError("fractional_order must be between zero and one")
        if not np.isfinite(self.impulse_threshold) or self.impulse_threshold < 0:
            raise ValueError("impulse_threshold must be finite and nonnegative")
        if type(self.hold_bars) is not int or not 3 <= self.hold_bars <= 5:
            raise ValueError("hold_bars must be an integer from three to five")


class GeneratedStrategy(BaseStrategy[GapMemoryParams]):
    strategy_id = "gen_local_1790880375"

    @classmethod
    def params_type(cls):
        return GapMemoryParams

    def warmup_bars(self, params: GapMemoryParams) -> int:
        return 8

    def indicators(self, data: pd.DataFrame, params: GapMemoryParams) -> pd.DataFrame:
        out = pd.DataFrame(index=data.index)
        close = data["close"].where(data["close"] > 0)
        out["log_close"] = np.log(close)
        # ponytail: six-tap fractional memory; longer kernels require relaxing the warmup cap.
        weights = [1.0]
        for k in range(1, 6):
            weights.append(-weights[-1] * (params.fractional_order - k + 1) / k)
        out["fractional_close"] = sum(
            weight * out["log_close"].shift(k) for k, weight in enumerate(weights)
        )
        out["impulse"] = out["fractional_close"].diff()
        out["prior_impulse"] = out["impulse"].shift(1)
        out["gap"] = data["open"] / close.shift(1) - 1.0
        out["prior_gap"] = out["gap"].shift(1)
        out["relative_range"] = (data["high"] - data["low"]) / close
        out["range_scale"] = out["relative_range"].rolling(5, min_periods=5).mean().shift(1)
        out["trigger"] = params.impulse_threshold * out["range_scale"]
        out = out.replace([np.inf, -np.inf], np.nan)
        out["entry"] = (
            (out["prior_gap"] < out["gap"])
            & (out["gap"] < 0)
            & (out["prior_impulse"] <= 0)
            & (out["impulse"] > out["trigger"])
            & (out["range_scale"] > 0)
            & out.notna().all(axis=1)
        ).fillna(False)
        return out

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: GapMemoryParams,
    ) -> SignalFrame:
        entries = indicators["entry"].fillna(False).to_numpy(dtype=bool)
        signal = np.zeros(len(data), dtype=int)
        exit_bar = -1
        for i in range(self.warmup_bars(params), len(data)):
            if i < exit_bar:
                signal[i] = 1
            elif i == exit_bar:
                # Force a flat bar even if another entry qualifies.
                signal[i] = 0
            elif entries[i]:
                signal[i] = 1
                exit_bar = i + params.hold_bars
        df = pd.DataFrame({"signal": signal, "size": 0.95}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


if __name__ == "__main__":
    index = pd.date_range("2020-01-01", periods=24, freq="B")
    close = 100.0 + np.sin(np.arange(24))
    data = pd.DataFrame(
        {"open": close, "high": close + 1, "low": close - 1, "close": close, "volume": 1000},
        index=index,
    )
    strategy = GeneratedStrategy()
    for holding in (3, 4, 5):
        params = GapMemoryParams(hold_bars=holding)
        ctx = StrategyContext(symbol="SPY", timeframe="1d", warmup_bars=8)
        entries = pd.DataFrame({"entry": False}, index=index)
        entries.iloc[8, 0] = True
        result = strategy.generate_signals(data, entries, ctx, params).data
        assert result["signal"].tolist() == [0] * 9 + [1] * holding + [0] * (15 - holding)
        entries.iloc[8:, 0] = True
        result = strategy.generate_signals(data, entries, ctx, params).data
        assert result["signal"].iloc[9:9 + holding].eq(1).all()
        assert result["signal"].iloc[9 + holding] == 0
        assert result["size"].gt(0).all()
    params = GapMemoryParams()
    full = strategy.indicators(data, params)
    assert full.iloc[:16].equals(strategy.indicators(data.iloc[:16], params))
