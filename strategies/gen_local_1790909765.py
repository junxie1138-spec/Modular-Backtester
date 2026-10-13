from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class Params:
    volume_lookback: int = 20
    pressure_bars: int = 4
    volume_ratio: float = 1.1
    min_pressure: float = 0.002


class GeneratedStrategy(BaseStrategy[Params]):
    strategy_id = "gen_local_1790909765"

    @classmethod
    def params_type(cls):
        return Params

    def warmup_bars(self, params: Params) -> int:
        return int(max(1, params.volume_lookback) + max(1, params.pressure_bars) + 1)

    def indicators(self, data: pd.DataFrame, params: Params) -> pd.DataFrame:
        volume_window = max(1, int(params.volume_lookback))
        pressure_window = max(1, int(params.pressure_bars))
        volume = data["volume"].astype(float).where(data["volume"].ge(0))
        close = data["close"].astype(float).where(data["close"].gt(0))
        baseline = volume.rolling(volume_window, min_periods=volume_window).mean().shift(1)
        relative_volume = volume.div(baseline.where(baseline.gt(0)))
        returns = close.pct_change(fill_method=None).replace([np.inf, -np.inf], np.nan)
        ready = relative_volume.notna() & returns.notna()
        confirmed_return = (returns * relative_volume).where(
            relative_volume.ge(max(0.0, params.volume_ratio)), 0.0
        ).where(ready)
        pressure = confirmed_return.rolling(pressure_window, min_periods=pressure_window).sum()
        result = pd.DataFrame(index=data.index)
        result["seasonal_window"] = data.index.dayofweek.isin([3, 4, 0, 1])
        result["volume_baseline"] = baseline
        result["relative_volume"] = relative_volume
        result["return"] = returns
        result["volume_confirmed_return"] = confirmed_return
        result["pressure"] = pressure
        result["ready"] = pressure.notna()
        return result.replace([np.inf, -np.inf], np.nan).fillna(0)

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: Params,
    ) -> SignalFrame:
        agreement = (
            indicators["ready"].astype(bool)
            & indicators["seasonal_window"].astype(bool)
            & indicators["pressure"].gt(max(0.0, params.min_pressure))
        )
        df = pd.DataFrame(index=data.index)
        # A held long exits only when the combined entry condition flips false.
        df["signal"] = agreement.astype(int)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = 0.95
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check() -> None:
    strategy = GeneratedStrategy()
    params = Params()
    index = pd.bdate_range("2024-01-04", periods=7)
    data = pd.DataFrame(
        {"open": 100.0, "high": 101.0, "low": 99.0, "close": 100.0, "volume": 1000.0},
        index=index,
    )
    indicators = pd.DataFrame(
        {
            "seasonal_window": index.dayofweek.isin([3, 4, 0, 1]),
            "pressure": [0.003, 0.003, 0.003, 0.003, 0.003, -0.003, 0.003],
            "ready": True,
        },
        index=index,
    )
    ctx = StrategyContext(symbol="SPY", timeframe="1d", warmup_bars=strategy.warmup_bars(params))
    signals = strategy.generate_signals(data, indicators, ctx, params).data
    assert signals["signal"].tolist() == [0, 1, 1, 1, 1, 0, 0]
    assert signals["size"].gt(0).all()
    warmup = strategy.indicators(data, params)
    assert not warmup["ready"].any()
    assert not strategy.generate_signals(data, warmup, ctx, params).data["signal"].any()


if __name__ == "__main__":
    _self_check()
