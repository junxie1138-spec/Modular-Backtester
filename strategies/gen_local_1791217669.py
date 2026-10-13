from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class Params:
    capacity: float = 6.0
    profit_target: float = 0.05

    def __post_init__(self):
        if not np.isfinite(self.capacity) or self.capacity <= 0:
            raise ValueError("capacity must be positive and finite")
        if not np.isfinite(self.profit_target) or not 0 < self.profit_target < 1:
            raise ValueError("profit_target must be between zero and one")


class GeneratedStrategy(BaseStrategy[Params]):
    strategy_id = "gen_local_1791217669"

    @classmethod
    def params_type(cls):
        return Params

    def warmup_bars(self, params: Params) -> int:
        return 83

    def indicators(self, data: pd.DataFrame, params: Params) -> pd.DataFrame:
        out = pd.DataFrame(index=data.index)
        out["return"] = data["close"].pct_change(fill_method=None)
        out["fast_vol"] = out["return"].rolling(5, min_periods=5).std()
        out["slow_vol"] = out["return"].rolling(63, min_periods=63).std()
        out["variance_load"] = (
            out["fast_vol"].pow(2) / out["slow_vol"].pow(2).replace(0, np.nan)
        ).replace([np.inf, -np.inf], np.nan) - 1.0
        out["backlog"] = out["variance_load"].rolling(20, min_periods=20).sum().clip(lower=0)
        out["momentum"] = data["close"].pct_change(20, fill_method=None).replace(
            [np.inf, -np.inf], np.nan
        )
        return out

    def generate_signals(
        self, data: pd.DataFrame, indicators: pd.DataFrame,
        ctx: StrategyContext, params: Params
    ) -> SignalFrame:
        close = data["close"].to_numpy(dtype=float)
        opening = data["open"].to_numpy(dtype=float)
        backlog = indicators["backlog"].to_numpy(dtype=float)
        momentum = indicators["momentum"].to_numpy(dtype=float)
        signal = np.zeros(len(data), dtype=int)
        stressed = False
        position = 0
        entry_price = np.nan
        entry_bar = -1
        for i in range(len(data)):
            event = 0
            if np.isfinite(backlog[i]):
                if not stressed and backlog[i] >= params.capacity:
                    stressed = True
                    event = 1
                elif stressed and backlog[i] <= params.capacity * 0.25:
                    stressed = False
                    event = -1
            if position:
                if np.isnan(entry_price):
                    entry_price = opening[i]
                    entry_bar = i
                if (
                    not np.isfinite(entry_price) or entry_price <= 0
                    or not np.isfinite(close[i]) or close[i] <= 0
                    or position * (close[i] / entry_price - 1.0) >= params.profit_target
                    or i - entry_bar + 1 >= 18
                ):
                    position = 0
                    entry_price = np.nan
                    continue
            elif (
                event and np.isfinite(momentum[i]) and momentum[i] != 0
                and np.isfinite(close[i]) and close[i] > 0
            ):
                position = event * (1 if momentum[i] > 0 else -1)
                entry_price = np.nan
            signal[i] = position
        df = pd.DataFrame({"signal": signal, "size": 1.0}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


if __name__ == "__main__":
    strategy = GeneratedStrategy.__new__(GeneratedStrategy)
    params = Params()
    index = pd.date_range("2020-01-01", periods=25, freq="B")
    for side in (-1, 1):
        for target in (False, True):
            data = pd.DataFrame(
                {"open": 100.0, "high": 110.0, "low": 90.0,
                 "close": 100.0, "volume": 1000.0}, index=index
            )
            if target:
                data.loc[index[3], "close"] = 100.0 * (1 + side * 0.06)
            derived = pd.DataFrame(
                {"backlog": 10.0, "momentum": float(side)}, index=index
            )
            result = strategy.generate_signals(data, derived, None, params).data
            exit_index = 4 if target else 19
            assert result["signal"].iloc[0] == 0
            assert (result["signal"].iloc[1:exit_index] == side).all()
            assert (result["signal"].iloc[exit_index:] == 0).all()
    derived["backlog"] = [10.0] + [0.0] * 24
    data["close"] = 100.0
    data.loc[index[1], "close"] = 106.0
    derived["momentum"] = 1.0
    result = strategy.generate_signals(data, derived, None, params).data
    assert result["signal"].iloc[2] == 0
