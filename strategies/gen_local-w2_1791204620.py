from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class VolatilityParams:
    baseline_bars: int = 2
    log_jump: float = 0.25
    size: float = 0.95


class GeneratedStrategy(BaseStrategy[VolatilityParams]):
    strategy_id = "gen_local-w2_1791204620"

    @classmethod
    def params_type(cls) -> type[VolatilityParams]:
        return VolatilityParams

    def warmup_bars(self, params: VolatilityParams) -> int:
        return int(params.baseline_bars + 2)

    def indicators(self, data: pd.DataFrame, params: VolatilityParams) -> pd.DataFrame:
        if params.baseline_bars < 1:
            raise ValueError("baseline_bars must be positive")
        if not np.isfinite(params.log_jump) or params.log_jump <= 0:
            raise ValueError("log_jump must be finite and positive")
        previous_close = data["close"].shift(1)
        previous_close = previous_close.where(previous_close > 0)
        components = pd.concat([
            data["high"] - data["low"],
            (data["high"] - previous_close).abs(),
            (data["low"] - previous_close).abs(),
        ], axis=1)
        true_range = components.max(axis=1, skipna=False)
        normalized_range = (true_range / previous_close).replace([np.inf, -np.inf], np.nan)
        normalized_range = normalized_range.where(normalized_range > 0)
        baseline_atr = normalized_range.shift(1).rolling(
            params.baseline_bars, min_periods=params.baseline_bars
        ).mean()
        log_change = np.log(normalized_range / baseline_atr.where(baseline_atr > 0))
        log_change = log_change.replace([np.inf, -np.inf], np.nan)
        return pd.DataFrame({
            "previous_close": previous_close,
            "true_range": true_range,
            "normalized_range": normalized_range,
            "baseline_atr": baseline_atr,
            "log_change": log_change,
        }, index=data.index)

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: VolatilityParams,
    ) -> SignalFrame:
        if not np.isfinite(params.log_jump) or params.log_jump <= 0:
            raise ValueError("log_jump must be finite and positive")
        if not np.isfinite(params.size) or params.size <= 0:
            raise ValueError("size must be finite and positive")
        signals = np.zeros(len(data), dtype=int)
        state = 0
        for i, change in enumerate(indicators["log_change"].to_numpy()):
            if np.isfinite(change):
                if change <= -params.log_jump:
                    state = 1
                elif change >= params.log_jump:
                    state = 0
            signals[i] = state
        df = pd.DataFrame({"signal": signals, "size": float(params.size)}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check() -> None:
    params = VolatilityParams()
    index = pd.date_range("2020-01-01", periods=7)
    data = pd.DataFrame({
        "open": 100.0, "high": 101.0, "low": 99.0,
        "close": 100.0, "volume": 1000.0,
    }, index=index)
    indicators = pd.DataFrame({
        "log_change": [np.nan, -0.4, 0.0, 0.4, -0.4, 0.4, np.nan]
    }, index=index)
    result = GeneratedStrategy.generate_signals(None, data, indicators, None, params)
    assert result.data["signal"].tolist() == [0, 0, 1, 1, 0, 1, 0]
    assert (result.data["size"] > 0).all()
    calculated = GeneratedStrategy.indicators(None, data, params)
    assert calculated["log_change"].iloc[:3].isna().all()
    assert np.allclose(calculated["log_change"].iloc[3:], 0.0)
    assert GeneratedStrategy.warmup_bars(None, params) == 4


if __name__ == "__main__":
    _self_check()
