from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class OrdinalPositionParams:
    window: int = 15
    edge: float = 0.10


class GeneratedStrategy(BaseStrategy[OrdinalPositionParams]):
    strategy_id = "gen_local-w2_1791300602"

    @classmethod
    def params_type(cls) -> type[OrdinalPositionParams]:
        return OrdinalPositionParams

    def warmup_bars(self, params: OrdinalPositionParams) -> int:
        return params.window + 2

    def indicators(self, data: pd.DataFrame, params: OrdinalPositionParams) -> pd.DataFrame:
        if params.window < 3 or not 0.0 <= params.edge < 1.0:
            raise ValueError("window must be at least 3 and edge must be in [0, 1)")
        close = data["close"].where(np.isfinite(data["close"]))
        prior = close.shift(1)
        earlier = close.shift(2)
        valid = close.notna() & prior.notna() & earlier.notna()
        pattern = (close.gt(prior).astype(int) + 2 * prior.gt(earlier).astype(int) + 4 * close.gt(earlier).astype(int)).where(valid)
        frequencies = pd.concat([
            pattern.eq(code).astype(float).where(valid).rolling(params.window, min_periods=params.window).mean()
            for code in range(8)
        ], axis=1)
        high = data["high"].rolling(params.window, min_periods=params.window).max()
        low = data["low"].rolling(params.window, min_periods=params.window).min()
        out = pd.DataFrame(index=data.index)
        out["complexity"] = (1.0 - frequencies.max(axis=1)).fillna(1.0)
        out["position"] = ((close - low) / (high - low).replace(0, np.nan)).replace([np.inf, -np.inf], np.nan).clip(0.0, 1.0).fillna(0.0)
        return out

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: OrdinalPositionParams) -> SignalFrame:
        # Exit only when the same relative-position condition that opened the trade flips.
        active = indicators["position"].gt(indicators["complexity"] + params.edge)
        df = pd.DataFrame({"signal": active.astype(int), "size": 1.0}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check() -> None:
    close = pd.Series(100.0 + np.arange(10), index=pd.date_range("2020-01-01", periods=10))
    data = pd.DataFrame({"open": close, "high": close, "low": close, "close": close, "volume": 1.0})
    strategy = GeneratedStrategy()
    params = OrdinalPositionParams(window=3)
    indicators = strategy.indicators(data, params)
    assert indicators["complexity"].iloc[5] == 0.0
    assert strategy.generate_signals(data, indicators, None, params).data["signal"].iloc[6] == 1
    indicators.iloc[7, indicators.columns.get_loc("position")] = 0.0
    signals = strategy.generate_signals(data, indicators, None, params).data["signal"]
    assert signals.iloc[7] == 1 and signals.iloc[8] == 0


if __name__ == "__main__":
    _self_check()
