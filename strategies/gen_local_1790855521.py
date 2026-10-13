from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class StrategyParams:
    impulse_mult: float = 1.2
    downside_cap: float = 0.5
    hold_bars: int = 8

    def __post_init__(self):
        if not np.isfinite(self.impulse_mult) or self.impulse_mult <= 0:
            raise ValueError("impulse_mult must be finite and positive")
        if not np.isfinite(self.downside_cap) or not 0 < self.downside_cap < 1:
            raise ValueError("downside_cap must be between zero and one")
        if not isinstance(self.hold_bars, int) or not 5 <= self.hold_bars <= 10:
            raise ValueError("hold_bars must be an integer from five to ten")


class GeneratedStrategy(BaseStrategy[StrategyParams]):
    strategy_id = "gen_local_1790855521"

    @classmethod
    def params_type(cls):
        return StrategyParams

    def warmup_bars(self, params: StrategyParams) -> int:
        return 10

    def indicators(self, data: pd.DataFrame, params: StrategyParams) -> pd.DataFrame:
        out = pd.DataFrame(index=data.index)
        close = data["close"].where(data["close"] > 0)
        out["return"] = close.pct_change(fill_method=None).replace([np.inf, -np.inf], np.nan)
        out["volatility"] = out["return"].rolling(8, min_periods=8).std(ddof=0)
        out["reference_volatility"] = out["volatility"].shift(1)
        out["up_square"] = out["return"].clip(lower=0).pow(2)
        out["down_square"] = out["return"].clip(upper=0).pow(2)
        out["up_rms"] = np.sqrt(out["up_square"].rolling(3, min_periods=3).mean())
        out["down_rms"] = np.sqrt(out["down_square"].rolling(3, min_periods=3).mean())
        out["entry"] = (
            (out["reference_volatility"] > 0)
            & (out["up_rms"] > params.impulse_mult * out["reference_volatility"])
            & (out["down_rms"] < params.downside_cap * out["reference_volatility"])
        ).fillna(False)
        return out

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: StrategyParams,
    ) -> SignalFrame:
        signal = np.zeros(len(data), dtype=int)
        exit_at = -1
        for i, enter in enumerate(indicators["entry"].fillna(False).to_numpy(dtype=bool)):
            if i < exit_at:
                signal[i] = 1
            elif i > exit_at and enter:
                signal[i] = 1
                exit_at = i + params.hold_bars
            # At exit_at, stay flat even if another entry condition is present.
        df = pd.DataFrame({"signal": signal, "size": 1.0}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


if __name__ == "__main__":
    strategy = GeneratedStrategy()
    params = StrategyParams()
    index = pd.date_range("2020-01-01", periods=32)
    data = pd.DataFrame(
        {"open": 100.0, "high": 101.0, "low": 99.0, "close": 100.0, "volume": 1000.0},
        index=index,
    )
    ctx = StrategyContext(symbol="SPY", timeframe="1d", warmup_bars=10)
    indicators = strategy.indicators(data, params)
    assert not indicators["entry"].any()
    assert indicators["reference_volatility"].iloc[:9].isna().all()
    indicators["entry"] = True
    result = strategy.generate_signals(data, indicators, ctx, params).data
    expected = [0] + [int(i % 9 < 8) for i in range(31)]
    assert result["signal"].tolist() == expected
    assert result["size"].gt(0).all()
