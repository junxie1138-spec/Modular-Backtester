from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class RecoveryParams:
    entry_z: float = 1.5
    hold_bars: int = 7


class GeneratedStrategy(BaseStrategy[RecoveryParams]):
    strategy_id = "gen_local_1790980891"

    @classmethod
    def params_type(cls):
        return RecoveryParams

    def warmup_bars(self, params: RecoveryParams) -> int:
        return 23

    def indicators(self, data: pd.DataFrame, params: RecoveryParams) -> pd.DataFrame:
        if not np.isfinite(params.entry_z) or params.entry_z <= 0:
            raise ValueError("entry_z must be finite and positive")
        close = data["close"].astype(float).replace([np.inf, -np.inf], np.nan)
        out = pd.DataFrame(index=data.index)
        out["ma"] = close.rolling(20, min_periods=20).mean()
        out["std"] = close.rolling(20, min_periods=20).std(ddof=0).replace(0.0, np.nan)
        out["z"] = ((close - out["ma"]) / out["std"]).replace([np.inf, -np.inf], np.nan)
        out["prior_z"] = out["z"].shift(1)
        out["reference_ma"] = out["ma"].shift(3)
        out["reference_std"] = out["std"].shift(3)
        out["frozen_z"] = ((close - out["reference_ma"]) / out["reference_std"]).replace([np.inf, -np.inf], np.nan)
        out["price_recovery"] = close.diff(3)
        out["mean_migration"] = (out["ma"] - out["reference_ma"]).abs()
        out["entry"] = (
            (out["prior_z"] < -params.entry_z)
            & (out["z"] >= -params.entry_z)
            & (out["frozen_z"] >= -params.entry_z)
            & (out["price_recovery"] > 4.0 * out["mean_migration"])
        ).fillna(False)
        return out

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: RecoveryParams,
    ) -> SignalFrame:
        if isinstance(params.hold_bars, bool) or not isinstance(params.hold_bars, (int, np.integer)) or params.hold_bars < 1:
            raise ValueError("hold_bars must be a positive integer")
        entry = indicators["entry"].fillna(False).to_numpy(dtype=bool)
        signal = np.zeros(len(data), dtype=int)
        exit_at = -1
        for i, eligible in enumerate(entry):
            if exit_at >= 0:
                if i < exit_at:
                    signal[i] = 1
                else:
                    exit_at = -1
            elif eligible:
                signal[i] = 1
                exit_at = i + params.hold_bars
        df = pd.DataFrame({"signal": signal, "size": 1.0}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check() -> None:
    strategy = GeneratedStrategy()
    params = RecoveryParams()
    index = pd.date_range("2020-01-01", periods=50, freq="B")
    data = pd.DataFrame({"open": 100.0, "high": 101.0, "low": 99.0, "close": 100.0, "volume": 1000.0}, index=index)
    ctx = StrategyContext(symbol="SPY", timeframe="1d", warmup_bars=23)
    indicators = strategy.indicators(data, params)
    assert not indicators["entry"].any()
    assert not strategy.generate_signals(data, indicators, ctx, params).data["signal"].any()
    indicators["entry"] = True
    signals = strategy.generate_signals(data, indicators, ctx, params).data
    assert signals["signal"].iloc[:9].tolist() == [0] + [1] * 7 + [0]
    assert signals["signal"].iloc[9:17].tolist() == [1] * 7 + [0]
    assert signals["size"].gt(0).all()


if __name__ == "__main__":
    _self_check()
