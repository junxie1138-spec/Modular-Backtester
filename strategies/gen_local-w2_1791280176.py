from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class Params:
    lookback: int = 30
    correlation_floor: float = 0.10
    minimum_daily_drift: float = 0.0002
    hold_bars: int = 7


class GeneratedStrategy(BaseStrategy[Params]):
    strategy_id = "gen_local-w2_1791280176"

    @classmethod
    def params_type(cls) -> type[Params]:
        return Params

    def warmup_bars(self, params: Params) -> int:
        if params.lookback < 3 or params.hold_bars < 1:
            raise ValueError("lookback must be >= 3 and hold_bars must be >= 1")
        if not 0.0 <= params.correlation_floor < 1.0:
            raise ValueError("correlation_floor must be in [0, 1)")
        if not np.isfinite(params.minimum_daily_drift) or params.minimum_daily_drift < 0.0:
            raise ValueError("minimum_daily_drift must be finite and nonnegative")
        return int(params.lookback + 4)

    def indicators(self, data: pd.DataFrame, params: Params) -> pd.DataFrame:
        self.warmup_bars(params)
        returns = data["close"].pct_change(fill_method=None).replace([np.inf, -np.inf], np.nan)
        lag_one = returns.shift(1)
        lag_two = returns.shift(2)
        window = returns.rolling(params.lookback, min_periods=params.lookback)
        drift = window.mean()
        correlation_one = window.corr(lag_one).replace([np.inf, -np.inf], np.nan)
        correlation_two = window.corr(lag_two).replace([np.inf, -np.inf], np.nan)
        qualified = (
            (drift > params.minimum_daily_drift)
            & (correlation_one > params.correlation_floor)
            & (correlation_two > params.correlation_floor)
            & (returns > 0.0)
        ).fillna(False)
        confirmed = qualified & qualified.shift(1, fill_value=False)
        return pd.DataFrame({
            "return": returns.fillna(0.0),
            "lag_one_return": lag_one.fillna(0.0),
            "lag_two_return": lag_two.fillna(0.0),
            "daily_drift": drift.fillna(0.0),
            "correlation_one": correlation_one.fillna(0.0),
            "correlation_two": correlation_two.fillna(0.0),
            "qualified": qualified,
            "confirmed": confirmed,
        }, index=data.index)

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame,
                         ctx: StrategyContext, params: Params) -> SignalFrame:
        self.warmup_bars(params)
        entries = indicators["confirmed"].fillna(False).to_numpy(dtype=bool)
        signals = np.zeros(len(data), dtype=int)
        exit_bar = -1
        for i in range(len(data)):
            if exit_bar >= 0:
                if i == exit_bar:
                    exit_bar = -1
                    continue
                signals[i] = 1
            elif entries[i]:
                signals[i] = 1
                exit_bar = i + params.hold_bars
        df = pd.DataFrame({"signal": signals, "size": 1.0}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


if __name__ == "__main__":
    strategy = GeneratedStrategy()
    params = Params()
    index = pd.date_range("2020-01-01", periods=24, freq="B")
    data = pd.DataFrame({"open": 100.0, "high": 101.0, "low": 99.0,
                         "close": 100.0, "volume": 1000.0}, index=index)
    indicators = pd.DataFrame({"confirmed": [False] + [True] * 23}, index=index)
    signal = strategy.generate_signals(data, indicators, None, params).data["signal"]
    assert signal.iloc[:2].eq(0).all()
    assert signal.iloc[2:9].eq(1).all() and signal.iloc[9] == 0
    assert signal.iloc[10:17].eq(1).all() and signal.iloc[17] == 0
    assert strategy.generate_signals(data, strategy.indicators(data, params), None, params).data["signal"].eq(0).all()
