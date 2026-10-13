from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class RecoveryParams:
    peak_bars: int = 63
    roc_bars: int = 5
    memory_bars: int = 18

    def __post_init__(self):
        if min(self.peak_bars, self.roc_bars, self.memory_bars) < 2:
            raise ValueError("All lookbacks must be at least two bars.")


class GeneratedStrategy(BaseStrategy[RecoveryParams]):
    strategy_id = "gen_local_1791325037"

    @classmethod
    def params_type(cls):
        return RecoveryParams

    def warmup_bars(self, params: RecoveryParams) -> int:
        return max(
            params.peak_bars + params.roc_bars + params.memory_bars,
            2 * params.roc_bars + params.memory_bars + 1,
        )

    def indicators(self, data: pd.DataFrame, params: RecoveryParams) -> pd.DataFrame:
        out = pd.DataFrame(index=data.index)
        close = data["close"].where(data["close"] > 0)
        out["peak"] = close.rolling(params.peak_bars).max()
        out["drawdown"] = 1.0 - close / out["peak"]
        out["roc"] = close / close.shift(params.roc_bars) - 1.0
        out["acceleration"] = out["roc"].diff(params.roc_bars) / params.roc_bars
        out["strain_weight"] = out["drawdown"].shift(params.roc_bars)
        out["weighted_acceleration"] = out["acceleration"] * out["strain_weight"]
        out["strain_sum"] = out["strain_weight"].rolling(params.memory_bars).sum()
        out["acceleration_sum"] = out["weighted_acceleration"].rolling(params.memory_bars).sum()
        out["recovery_score"] = out["acceleration_sum"] / out["strain_sum"].where(out["strain_sum"] > 0)
        return out.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: RecoveryParams,
    ) -> SignalFrame:
        df = pd.DataFrame(index=data.index)
        # The same zero boundary opens and closes exposure; no timer or stop.
        df["signal"] = indicators["recovery_score"].gt(0.0).astype(int)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = 0.95
        return SignalFrame(data=df, signal_column="signal", size_column="size")


if __name__ == "__main__":
    params = RecoveryParams()
    strategy = GeneratedStrategy()
    index = pd.date_range("2020-01-01", periods=320, freq="B")
    close = 100.0 * np.exp(0.04 * np.sin(np.arange(320) / 12.0))
    data = pd.DataFrame({k: close for k in ("open", "high", "low", "close")}, index=index)
    data["volume"] = 1000000.0
    derived = strategy.indicators(data, params)
    signals = strategy.generate_signals(data, derived, None, params).data
    expected = derived["recovery_score"].gt(0).astype(int).shift(1).fillna(0).astype(int)
    assert signals["signal"].equals(expected)
    assert set(signals["signal"]) == {0, 1} and signals["size"].gt(0).all()
    assert np.isfinite(derived.to_numpy()).all()
    assert strategy.indicators(data.iloc[:200], params).equals(derived.iloc[:200])
    assert signals.iloc[:strategy.warmup_bars(params) - 1]["signal"].eq(0).all()
