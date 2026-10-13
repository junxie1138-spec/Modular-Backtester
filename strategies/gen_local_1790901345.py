from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class CoolingParams:
    block_bars: int = 4
    shock_ratio: float = 1.25
    cooling_ratio: float = 0.80


class GeneratedStrategy(BaseStrategy[CoolingParams]):
    strategy_id = "gen_local_1790901345"

    @classmethod
    def params_type(cls) -> type[CoolingParams]:
        return CoolingParams

    def warmup_bars(self, params: CoolingParams) -> int:
        return int(max(200, 21 + 2 * params.block_bars))

    def indicators(self, data: pd.DataFrame, params: CoolingParams) -> pd.DataFrame:
        if params.block_bars < 1:
            raise ValueError("block_bars must be positive")
        if not np.isfinite(params.shock_ratio) or params.shock_ratio <= 1.0:
            raise ValueError("shock_ratio must exceed one")
        if not np.isfinite(params.cooling_ratio) or not 0.0 < params.cooling_ratio < 1.0:
            raise ValueError("cooling_ratio must lie between zero and one")

        n = params.block_bars
        out = pd.DataFrame(index=data.index)
        out["ma200"] = data["close"].rolling(200, min_periods=200).mean()
        out["previous_close"] = data["close"].shift(1)
        out["true_range"] = pd.concat([
            data["high"] - data["low"],
            (data["high"] - out["previous_close"]).abs(),
            (data["low"] - out["previous_close"]).abs(),
        ], axis=1).max(axis=1)
        denominator = out["previous_close"].where(out["previous_close"] > 0)
        out["normalized_range"] = out["true_range"] / denominator
        out["recent_atr"] = out["normalized_range"].rolling(n, min_periods=n).mean()
        out["prior_atr"] = out["recent_atr"].shift(n)
        out["baseline_atr"] = out["normalized_range"].rolling(20, min_periods=20).mean().shift(2 * n)
        out["shock_ratio"] = out["prior_atr"] / out["baseline_atr"].where(out["baseline_atr"] > 0)
        out["cooling_ratio"] = out["recent_atr"] / out["prior_atr"].where(out["prior_atr"] > 0)
        out["recovery"] = data["close"] / data["close"].shift(n).where(data["close"].shift(n) > 0) - 1.0
        return out.replace([np.inf, -np.inf], np.nan)

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: CoolingParams,
    ) -> SignalFrame:
        required = ["ma200", "shock_ratio", "cooling_ratio", "recovery"]
        ready = indicators[required].notna().all(axis=1)
        entry_condition = (
            ready
            & (data["close"] > indicators["ma200"])
            & (indicators["shock_ratio"] >= params.shock_ratio)
            & (indicators["cooling_ratio"] <= params.cooling_ratio)
            & (indicators["recovery"] > 0.0)
        )
        df = pd.DataFrame(index=data.index)
        # Maintain exposure while the entry condition holds; its reversal exits.
        df["signal"] = entry_condition.fillna(False).astype(int)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = 0.95
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check() -> None:
    strategy = GeneratedStrategy.__new__(GeneratedStrategy)
    params = CoolingParams()
    index = pd.date_range("2020-01-01", periods=240, freq="D")
    data = pd.DataFrame({"open": 100.0, "high": 101.0, "low": 99.0, "close": 100.0, "volume": 1000.0}, index=index)
    indicators = strategy.indicators(data, params)
    assert indicators.index.equals(data.index)
    assert strategy.warmup_bars(params) >= 200
    assert not strategy.generate_signals(data, indicators, None, params).data["signal"].any()
    indicators["ma200"] = 99.0
    indicators["shock_ratio"] = 1.5
    indicators["recovery"] = 0.01
    indicators.iloc[210:214, indicators.columns.get_loc("cooling_ratio")] = 0.6
    result = strategy.generate_signals(data, indicators, None, params).data
    expected = pd.Series(0, index=index, dtype=int, name="signal")
    expected.iloc[211:215] = 1
    assert result["signal"].equals(expected)
    assert (result["size"] > 0).all()


if __name__ == "__main__":
    _self_check()
