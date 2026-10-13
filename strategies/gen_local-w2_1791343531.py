from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class DriftParams:
    lookback: int = 10
    entry_strength: float = 0.20
    minimum_size: float = 0.20
    maximum_size: float = 0.95


class GeneratedStrategy(BaseStrategy[DriftParams]):
    strategy_id = "gen_local-w2_1791343531"

    @classmethod
    def params_type(cls) -> type[DriftParams]:
        return DriftParams

    def warmup_bars(self, params: DriftParams) -> int:
        return int(params.lookback) + 1

    def indicators(self, data: pd.DataFrame, params: DriftParams) -> pd.DataFrame:
        if params.lookback < 2 or not 0.0 < params.entry_strength < 1.0:
            raise ValueError("lookback must be >= 2 and entry_strength must be between 0 and 1")
        if not 0.0 < params.minimum_size <= params.maximum_size <= 1.0:
            raise ValueError("sizes must satisfy 0 < minimum_size <= maximum_size <= 1")
        close = data["close"].astype(float).where(data["close"] > 0)
        returns = close.pct_change(fill_method=None).replace([np.inf, -np.inf], np.nan)
        window = returns.rolling(params.lookback, min_periods=params.lookback)
        net_return = window.sum()
        gross_return = returns.abs().rolling(params.lookback, min_periods=params.lookback).sum()
        largest_gain = returns.clip(lower=0.0).rolling(params.lookback, min_periods=params.lookback).max()
        residual_net = net_return - largest_gain
        residual_gross = gross_return - largest_gain
        strength = (residual_net / residual_gross.where(residual_gross > 1e-12)).clip(-1.0, 1.0)
        return pd.DataFrame({
            "return": returns,
            "net_return": net_return,
            "gross_return": gross_return,
            "largest_gain": largest_gain,
            "residual_net": residual_net,
            "residual_gross": residual_gross,
            "strength": strength,
        }, index=data.index).replace([np.inf, -np.inf], np.nan).fillna(0.0)

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: DriftParams) -> SignalFrame:
        strength = indicators["strength"].fillna(0.0)
        entry_condition = strength > params.entry_strength
        conviction = ((strength - params.entry_strength) / (1.0 - params.entry_strength)).clip(0.0, 1.0)
        df = pd.DataFrame(index=data.index)
        # The position exits only when this same entry condition becomes false.
        df["signal"] = entry_condition.astype(int)
        df["size"] = params.minimum_size + (params.maximum_size - params.minimum_size) * conviction
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = df["size"].shift(1).fillna(params.minimum_size).astype(float)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check() -> None:
    params = DriftParams()
    returns = np.full(45, 0.002)
    returns[20] = 0.15
    returns[21:31] = -0.002
    close = 100.0 * np.cumprod(1.0 + returns)
    data = pd.DataFrame({"open": close, "high": close, "low": close, "close": close, "volume": 1000.0}, index=pd.date_range("2020-01-01", periods=len(close)))
    indicators = GeneratedStrategy.indicators(None, data, params)
    result = GeneratedStrategy.generate_signals(None, data, indicators, None, params).data
    expected = (indicators["strength"] > params.entry_strength).shift(1).fillna(False).astype(int)
    assert result["signal"].equals(expected)
    assert result["signal"].iloc[11] == 1 and result["signal"].iloc[31] == 0
    assert indicators["strength"].iloc[29] < 0.0
    assert result["size"].between(params.minimum_size, params.maximum_size).all()
    assert np.isfinite(indicators.to_numpy()).all()
    prefix = GeneratedStrategy.indicators(None, data.iloc[:25], params)
    assert prefix.equals(indicators.iloc[:25])


if __name__ == "__main__":
    _self_check()
