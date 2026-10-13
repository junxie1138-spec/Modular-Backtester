from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class ExpirationParams:
    entry_z: float = 1.5
    holding_bars: int = 18


class GeneratedStrategy(BaseStrategy[ExpirationParams]):
    strategy_id = "gen_local-w2_1791320042"

    @classmethod
    def params_type(cls) -> type[ExpirationParams]:
        return ExpirationParams

    def warmup_bars(self, params: ExpirationParams) -> int:
        return 63

    def indicators(self, data: pd.DataFrame, params: ExpirationParams) -> pd.DataFrame:
        if not np.isfinite(params.entry_z) or params.entry_z <= 0:
            raise ValueError("entry_z must be finite and positive")
        if isinstance(params.holding_bars, bool) or not isinstance(params.holding_bars, int) or params.holding_bars < 1:
            raise ValueError("holding_bars must be a positive integer")
        close = data["close"].astype(float).replace([np.inf, -np.inf], np.nan)
        out = pd.DataFrame(index=data.index)
        out["ma"] = close.rolling(63, min_periods=63).mean()
        out["std"] = close.rolling(63, min_periods=63).std(ddof=0)
        out["z"] = ((close - out["ma"]) / out["std"].replace(0.0, np.nan)).replace([np.inf, -np.inf], np.nan)
        index = data.index
        out["expiration"] = (index.dayofweek == 4) & (index.day >= 15) & (index.day <= 21)
        valid = out["expiration"] & out["z"].notna()
        out["entry"] = np.select(
            [valid & (out["z"] <= -params.entry_z), valid & (out["z"] >= params.entry_z)],
            [1, -1], default=0,
        ).astype(int)
        return out

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: ExpirationParams) -> SignalFrame:
        if isinstance(params.holding_bars, bool) or not isinstance(params.holding_bars, int) or params.holding_bars < 1:
            raise ValueError("holding_bars must be a positive integer")
        entries = indicators["entry"].fillna(0).to_numpy(dtype=int)
        signals = np.zeros(len(data), dtype=int)
        position, exit_bar = 0, -1
        for i, entry in enumerate(entries):
            if position and i == exit_bar:
                position = 0
            elif not position and entry:
                position = int(entry)
                exit_bar = i + params.holding_bars
            signals[i] = position
        df = pd.DataFrame({"signal": signals, "size": 1.0}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check() -> None:
    strategy = GeneratedStrategy()
    params = ExpirationParams()
    index = pd.date_range("2022-01-21", periods=80, freq="B")
    data = pd.DataFrame({"open": 100.0, "high": 101.0, "low": 99.0, "close": 100.0, "volume": 1000.0}, index=index)
    derived = strategy.indicators(data, params)
    assert derived.index.equals(data.index)
    assert not derived["entry"].any()
    entries = np.zeros(len(data), dtype=int)
    entries[0], entries[1], entries[19] = 1, -1, -1
    result = strategy.generate_signals(data, pd.DataFrame({"entry": entries}, index=index), None, params).data
    assert result["signal"].iloc[0] == 0
    assert (result["signal"].iloc[1:19] == 1).all()
    assert result["signal"].iloc[19] == 0
    assert (result["signal"].iloc[20:38] == -1).all()
    assert (result["signal"].iloc[38:] == 0).all()
    assert (result["size"] > 0).all()


if __name__ == "__main__":
    _self_check()
