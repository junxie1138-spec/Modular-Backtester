from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class VolatilityMomentumParams:
    volatility_window: int = 20
    momentum_window: int = 10
    upside_ratio: float = 1.4
    holding_bars: int = 18


class GeneratedStrategy(BaseStrategy[VolatilityMomentumParams]):
    strategy_id = "gen_local_1791277642"

    @classmethod
    def params_type(cls) -> type[VolatilityMomentumParams]:
        return VolatilityMomentumParams

    def warmup_bars(self, params: VolatilityMomentumParams) -> int:
        return int(max(params.volatility_window, params.momentum_window) + 1)

    def indicators(self, data: pd.DataFrame, params: VolatilityMomentumParams) -> pd.DataFrame:
        if params.volatility_window < 2 or params.momentum_window < 1:
            raise ValueError("Indicator windows must be positive; volatility requires at least two bars.")
        if params.holding_bars < 1 or not np.isfinite(params.upside_ratio) or params.upside_ratio <= 1.0:
            raise ValueError("Holding bars must be positive and upside_ratio must exceed one.")
        result = pd.DataFrame(index=data.index)
        close = data["close"].replace(0.0, np.nan)
        result["return"] = close.pct_change(fill_method=None).replace([np.inf, -np.inf], np.nan)
        result["upside_return"] = result["return"].clip(lower=0.0)
        result["downside_return"] = result["return"].clip(upper=0.0)
        result["upside_std"] = result["upside_return"].rolling(params.volatility_window, min_periods=params.volatility_window).std(ddof=0)
        result["downside_std"] = result["downside_return"].rolling(params.volatility_window, min_periods=params.volatility_window).std(ddof=0)
        result["momentum"] = close.pct_change(periods=params.momentum_window, fill_method=None).replace([np.inf, -np.inf], np.nan)
        return result

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: VolatilityMomentumParams) -> SignalFrame:
        if params.holding_bars < 1:
            raise ValueError("holding_bars must be positive.")
        ready = np.isfinite(indicators[["upside_std", "downside_std", "momentum"]]).all(axis=1)
        volatility_agrees = indicators["upside_std"] > params.upside_ratio * indicators["downside_std"]
        momentum_agrees = indicators["momentum"] > 0.0
        eligible = (ready & volatility_agrees & momentum_agrees).fillna(False).to_numpy(dtype=bool)
        signal = np.zeros(len(data), dtype=int)
        entry_bar = -1
        for bar in range(len(data)):
            if entry_bar >= 0:
                if bar - entry_bar == params.holding_bars:
                    entry_bar = -1
                else:
                    signal[bar] = 1
            elif eligible[bar]:
                entry_bar = bar
                signal[bar] = 1
        df = pd.DataFrame({"signal": signal, "size": 0.95}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


if __name__ == "__main__":
    params = VolatilityMomentumParams(holding_bars=3)
    index = pd.date_range("2020-01-01", periods=10)
    data = pd.DataFrame(index=index)
    indicators = pd.DataFrame({"upside_std": 0.02, "downside_std": 0.01, "momentum": 0.03}, index=index)
    result = GeneratedStrategy.generate_signals(None, data, indicators, None, params).data
    assert result["signal"].tolist() == [0, 1, 1, 1, 0, 1, 1, 1, 0, 1]
    indicators.loc[index[1]:, "momentum"] = -0.03
    result = GeneratedStrategy.generate_signals(None, data, indicators, None, params).data
    assert result["signal"].tolist() == [0, 1, 1, 1, 0, 0, 0, 0, 0, 0]
    indicators.loc[:, "upside_std"] = np.nan
    result = GeneratedStrategy.generate_signals(None, data, indicators, None, params).data
    assert result["signal"].eq(0).all() and result["size"].gt(0).all()
