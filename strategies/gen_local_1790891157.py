from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class RecoveryParams:
    lookback: int = 6
    entry_z: float = 1.3
    recovery_z: float = 0.35
    hold_bars: int = 4


class GeneratedStrategy(BaseStrategy[RecoveryParams]):
    strategy_id = "gen_local_1790891157"

    @classmethod
    def params_type(cls):
        return RecoveryParams

    def warmup_bars(self, params: RecoveryParams) -> int:
        if not 3 <= params.lookback <= 9 or not 3 <= params.hold_bars <= 5:
            raise ValueError("lookback must be 3..9 and hold_bars must be 3..5")
        if not params.entry_z > 0 or not params.recovery_z > 0:
            raise ValueError("z-score thresholds must be positive")
        return int(params.lookback + 1)

    def indicators(self, data: pd.DataFrame, params: RecoveryParams) -> pd.DataFrame:
        self.warmup_bars(params)
        close = data["close"]
        window = close.rolling(params.lookback, min_periods=params.lookback)
        out = pd.DataFrame(index=data.index)
        out["ma"] = window.mean().shift(1)
        out["sd"] = window.std(ddof=0).shift(1).replace(0.0, np.nan)
        out["prior_z"] = (close.shift(1) - out["ma"]) / out["sd"]
        out["current_z"] = (close - out["ma"]) / out["sd"]
        out["recovery_z"] = out["current_z"] - out["prior_z"]
        out["low_sweep"] = data["low"].lt(data["low"].shift(1))
        return out.replace([np.inf, -np.inf], np.nan)

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: RecoveryParams,
    ) -> SignalFrame:
        self.warmup_bars(params)
        entries = (
            indicators["prior_z"].le(-params.entry_z)
            & indicators["current_z"].lt(0.0)
            & indicators["recovery_z"].ge(params.recovery_z)
            & indicators["low_sweep"].fillna(False)
        ).fillna(False).to_numpy(dtype=bool)
        signal = np.zeros(len(data), dtype=int)
        exit_bar = -1
        for i, enter in enumerate(entries):
            if exit_bar >= 0:
                if i < exit_bar:
                    signal[i] = 1
                else:
                    exit_bar = -1
            elif enter:
                signal[i] = 1
                exit_bar = i + params.hold_bars
        df = pd.DataFrame({"signal": signal, "size": 0.95}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


if __name__ == "__main__":
    strategy = GeneratedStrategy()
    index = pd.date_range("2020-01-01", periods=24)
    data = pd.DataFrame(
        {"open": 100.0, "high": 101.0, "low": 99.0, "close": 100.0, "volume": 1000.0},
        index=index,
    )
    forced = pd.DataFrame(
        {"prior_z": -2.0, "current_z": -1.0, "recovery_z": 1.0, "low_sweep": True},
        index=index,
    )
    for hold in (3, 4, 5):
        params = RecoveryParams(hold_bars=hold)
        ctx = StrategyContext("SPY", "1d", strategy.warmup_bars(params))
        result = strategy.generate_signals(data, forced, ctx, params).data
        bars = np.arange(len(data))
        expected = ((bars > 0) & ((bars - 1) % (hold + 1) < hold)).astype(int)
        assert np.array_equal(result["signal"].to_numpy(), expected)
        assert result["size"].gt(0).all()
        flat = strategy.indicators(data, params)
        assert strategy.generate_signals(data, flat, ctx, params).data["signal"].eq(0).all()
    assert strategy.warmup_bars(RecoveryParams(lookback=9)) == 10
