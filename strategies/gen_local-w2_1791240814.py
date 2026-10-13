from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class FractionalEfficiencyParams:
    fractional_order: float = 0.45
    entry_strength: float = 0.70
    reset_strength: float = 0.25
    holding_bars: int = 4


class GeneratedStrategy(BaseStrategy[FractionalEfficiencyParams]):
    strategy_id = "gen_local-w2_1791240814"

    @classmethod
    def params_type(cls) -> type[FractionalEfficiencyParams]:
        return FractionalEfficiencyParams

    def warmup_bars(self, params: FractionalEfficiencyParams) -> int:
        return 7

    def indicators(self, data: pd.DataFrame, params: FractionalEfficiencyParams) -> pd.DataFrame:
        if not 0.0 < params.fractional_order < 1.0:
            raise ValueError("fractional_order must be between zero and one")
        if not 0.0 <= params.reset_strength < params.entry_strength <= 1.0:
            raise ValueError("Require 0 <= reset_strength < entry_strength <= 1")
        if params.holding_bars not in (3, 4, 5):
            raise ValueError("holding_bars must be 3, 4, or 5")

        out = pd.DataFrame(index=data.index)
        out["close"] = data["close"].astype(float).replace([np.inf, -np.inf], np.nan)
        out["change"] = out["close"].diff()
        weights = np.ones(6, dtype=float)
        for k in range(1, 6):
            weights[k] = weights[k - 1] * (k - 1 - params.fractional_order) / k

        fractional = pd.Series(0.0, index=data.index)
        travel = pd.Series(0.0, index=data.index)
        cumulative = np.cumsum(weights)
        for k in range(6):
            fractional = fractional + weights[k] * out["close"].shift(k)
        fractional = fractional - weights.sum() * out["close"].shift(5)
        for k in range(5):
            travel = travel + cumulative[k] * out["change"].shift(k).abs()

        out["fractional_close"] = fractional
        out["weighted_travel"] = travel
        out["strength"] = (fractional / travel.where(travel > 0.0)).replace([np.inf, -np.inf], np.nan).fillna(0.0).clip(-1.0, 1.0)
        out["direction"] = np.sign(out["strength"]).astype(int)
        out["eligible"] = out["strength"].abs().ge(params.entry_strength)
        out["confirmed"] = out["eligible"] & out["eligible"].shift(1, fill_value=False) & out["direction"].eq(out["direction"].shift(1))
        out["reset"] = out["weighted_travel"].notna() & out["strength"].abs().le(params.reset_strength)
        return out

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: FractionalEfficiencyParams) -> SignalFrame:
        signals = np.zeros(len(data), dtype=int)
        confirmed = indicators["confirmed"].to_numpy(dtype=bool)
        reset = indicators["reset"].to_numpy(dtype=bool)
        direction = indicators["direction"].to_numpy(dtype=int)
        position = 0
        exit_bar = -1
        armed = True

        for i in range(len(data)):
            if position != 0:
                if i == exit_bar:
                    position = 0
                    continue
                signals[i] = position
                continue
            if reset[i]:
                armed = True
            if armed and confirmed[i]:
                position = int(direction[i])
                exit_bar = i + params.holding_bars
                armed = False
                signals[i] = position

        df = pd.DataFrame(index=data.index)
        df["signal"] = signals
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = 1.0
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check() -> None:
    strategy = GeneratedStrategy()
    params = FractionalEfficiencyParams()
    index = pd.date_range("2020-01-01", periods=14)
    data = pd.DataFrame({"open": 100.0, "high": 101.0, "low": 99.0, "close": 100.0, "volume": 1000.0}, index=index)
    flat = strategy.indicators(data, params)
    assert flat["strength"].eq(0.0).all()
    data["close"] = np.arange(100.0, 114.0)
    rising = strategy.indicators(data, params)
    assert np.allclose(rising["strength"].iloc[5:], 1.0)
    assert not rising["confirmed"].iloc[:6].any()
    probe = pd.DataFrame({"confirmed": True, "reset": False, "direction": 1}, index=index)
    result = strategy.generate_signals(data, probe, None, params).data
    assert result["signal"].tolist() == [0, 1, 1, 1, 1] + [0] * 9
    probe["direction"] = -1
    result = strategy.generate_signals(data, probe, None, params).data
    assert result["signal"].tolist() == [0, -1, -1, -1, -1] + [0] * 9


if __name__ == "__main__":
    _self_check()
