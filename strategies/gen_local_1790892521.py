from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class RecoveryParams:
    peak_lookback: int = 63
    min_depth: float = 0.04
    recovery_fraction: float = 0.06
    holding_bars: int = 8


class GeneratedStrategy(BaseStrategy[RecoveryParams]):
    strategy_id = "gen_local_1790892521"

    @classmethod
    def params_type(cls) -> type[RecoveryParams]:
        return RecoveryParams

    def warmup_bars(self, params: RecoveryParams) -> int:
        return max(2, params.peak_lookback) + 2

    def indicators(self, data: pd.DataFrame, params: RecoveryParams) -> pd.DataFrame:
        out = pd.DataFrame(index=data.index)
        close = data["close"].astype(float)
        out["peak"] = close.rolling(max(2, params.peak_lookback), min_periods=max(2, params.peak_lookback)).max()
        out["depth"] = 1.0 - close / out["peak"].where(out["peak"] > 0.0)
        previous_depth = out["depth"].shift(1)
        out["release"] = (previous_depth - out["depth"]) / previous_depth.where(previous_depth > 0.0)
        out["recovery_step"] = (
            np.isfinite(out["release"])
            & (out["release"] >= max(0.0, params.recovery_fraction))
            & (out["depth"] < previous_depth)
            & (out["peak"] == out["peak"].shift(1))
        ).fillna(False)
        out["confirmed"] = (
            out["recovery_step"]
            & out["recovery_step"].shift(1, fill_value=False)
            & (out["depth"] >= max(0.0, params.min_depth))
            & np.isfinite(out["depth"])
        ).fillna(False)
        return out

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: RecoveryParams) -> SignalFrame:
        entries = indicators["confirmed"].fillna(False).to_numpy(dtype=bool)
        signals = np.zeros(len(data), dtype=int)
        holding_bars = max(1, int(params.holding_bars))
        exit_bar = -1
        for i in range(len(data)):
            if exit_bar >= 0:
                if i == exit_bar:
                    exit_bar = -1
                else:
                    signals[i] = 1
            elif entries[i]:
                signals[i] = 1
                exit_bar = i + holding_bars
        df = pd.DataFrame(index=data.index)
        df["signal"] = signals
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = 0.95
        return SignalFrame(data=df, signal_column="signal", size_column="size")


if __name__ == "__main__":
    closes = [100.0] * 6 + [90.0, 91.0, 92.0, 93.0, 94.0, 95.0, 96.0, 97.0]
    sample = pd.DataFrame({"open": closes, "high": closes, "low": closes, "close": closes, "volume": 1000.0}, index=pd.date_range("2020-01-01", periods=len(closes)))
    params = RecoveryParams(peak_lookback=6, min_depth=0.03, recovery_fraction=0.05, holding_bars=3)
    strategy = GeneratedStrategy()
    derived = strategy.indicators(sample, params)
    result = strategy.generate_signals(sample, derived, None, params).data
    assert not derived["confirmed"].iloc[7]
    assert derived["confirmed"].iloc[8]
    assert result["signal"].tolist() == [0] * 9 + [1] * 3 + [0] * 2
    assert result["size"].gt(0.0).all()
