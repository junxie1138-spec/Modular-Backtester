from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class Params:
    fractional_order: float = 0.45
    tension_threshold: float = 0.9
    holding_bars: int = 4


class GeneratedStrategy(BaseStrategy[Params]):
    strategy_id = "gen_local_1791285921"

    @classmethod
    def params_type(cls) -> type[Params]:
        return Params

    @staticmethod
    def warmup_bars(params: Params) -> int:
        return 9

    @staticmethod
    def indicators(data: pd.DataFrame, params: Params) -> pd.DataFrame:
        if not np.isfinite(params.fractional_order) or not 0.0 < params.fractional_order < 1.0:
            raise ValueError("fractional_order must be between zero and one")
        if not np.isfinite(params.tension_threshold) or params.tension_threshold <= 0.0:
            raise ValueError("tension_threshold must be positive")
        if not isinstance(params.holding_bars, int) or isinstance(params.holding_bars, bool) or not 3 <= params.holding_bars <= 5:
            raise ValueError("holding_bars must be an integer from three through five")
        close = data["close"].astype(float)
        close = close.where(np.isfinite(close) & close.gt(0.0))
        fractional_close = close.copy()
        weight = 1.0
        # ponytail: five-tap fractional approximation; extend only if the warmup ceiling increases.
        for lag in range(1, 5):
            weight *= (lag - 1 - params.fractional_order) / lag
            fractional_close = fractional_close + weight * close.shift(lag)
        reference = fractional_close.rolling(5, min_periods=5)
        mean = reference.mean()
        std = reference.std(ddof=0).replace(0.0, np.nan)
        tension = ((fractional_close - mean) / std).replace([np.inf, -np.inf], np.nan)
        return pd.DataFrame({
            "fractional_close": fractional_close,
            "fractional_mean": mean,
            "fractional_std": std,
            "tension": tension,
            "is_monday": data.index.dayofweek == 0,
        }, index=data.index)

    @staticmethod
    def generate_signals(data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: Params) -> SignalFrame:
        if not isinstance(params.holding_bars, int) or isinstance(params.holding_bars, bool) or not 3 <= params.holding_bars <= 5:
            raise ValueError("holding_bars must be an integer from three through five")
        tension = indicators["tension"]
        entries = (
            indicators["is_monday"].fillna(False)
            & tension.notna()
            & np.isfinite(tension)
            & tension.le(-params.tension_threshold)
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
                exit_bar = i + params.holding_bars
        df = pd.DataFrame({"signal": signal, "size": 0.95}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


if __name__ == "__main__":
    index = pd.date_range("2024-01-01", periods=20, freq="B")
    data = pd.DataFrame({"open": 100.0, "high": 101.0, "low": 99.0, "close": np.arange(20, dtype=float) + 100.0, "volume": 1000.0}, index=index)
    strategy = GeneratedStrategy()
    actual = strategy.indicators(data, Params())
    assert actual["tension"].first_valid_index() == index[8]
    forced = pd.DataFrame({"tension": -2.0, "is_monday": index.dayofweek == 0}, index=index)
    for hold in (3, 4, 5):
        result = strategy.generate_signals(data, forced, None, Params(holding_bars=hold)).data
        assert result["signal"].iloc[0] == 0
        assert result["signal"].iloc[1:hold + 1].eq(1).all()
        assert result["signal"].iloc[hold + 1] == 0
        assert result["signal"].isin([0, 1]).all()
        assert result["size"].gt(0).all()
