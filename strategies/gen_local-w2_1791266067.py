from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class RepairParams:
    debt_window: int = 12
    roc_bars: int = 3
    acceleration_bars: int = 2
    projection_bars: int = 4
    repair_fraction: float = 0.5
    minimum_debt: float = 0.0075


class GeneratedStrategy(BaseStrategy[RepairParams]):
    strategy_id = "gen_local-w2_1791266067"

    @classmethod
    def params_type(cls) -> type[RepairParams]:
        return RepairParams

    def warmup_bars(self, params: RepairParams) -> int:
        return int(max(params.debt_window, params.roc_bars + params.acceleration_bars + 1) + 1)

    def indicators(self, data: pd.DataFrame, params: RepairParams) -> pd.DataFrame:
        close = data["close"].astype(float).where(data["close"] > 0)
        out = pd.DataFrame(index=data.index)
        out["peak"] = close.rolling(params.debt_window, min_periods=params.debt_window).max()
        out["trough"] = close.rolling(params.debt_window, min_periods=params.debt_window).min()
        out["drawdown_debt"] = out["peak"] / close - 1.0
        out["rebound_debt"] = 1.0 - out["trough"] / close
        out["roc"] = close.pct_change(periods=params.roc_bars, fill_method=None) / params.roc_bars
        out["roc_change"] = out["roc"].diff()
        out["acceleration"] = out["roc_change"].rolling(params.acceleration_bars, min_periods=params.acceleration_bars).mean()
        horizon = params.projection_bars
        out["projected_return"] = horizon * out["roc"] + 0.5 * horizon * (horizon + 1) * out["acceleration"]
        out = out.replace([np.inf, -np.inf], np.nan)
        out["long_entry"] = (
            (out["drawdown_debt"] >= params.minimum_debt)
            & (out["acceleration"] > 0.0)
            & (out["projected_return"] >= params.repair_fraction * out["drawdown_debt"])
        ).fillna(False)
        out["short_entry"] = (
            (out["rebound_debt"] >= params.minimum_debt)
            & (out["acceleration"] < 0.0)
            & (-out["projected_return"] >= params.repair_fraction * out["rebound_debt"])
        ).fillna(False)
        out["entry_direction"] = np.select([out["long_entry"], out["short_entry"]], [1, -1], default=0).astype(int)
        out["confirmed_direction"] = out["entry_direction"].where(
            out["entry_direction"].eq(out["entry_direction"].shift(1)), 0
        ).astype(int)
        return out

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: RepairParams,
    ) -> SignalFrame:
        raw = indicators["entry_direction"].fillna(0).to_numpy(dtype=int)
        confirmed = indicators["confirmed_direction"].fillna(0).to_numpy(dtype=int)
        positions = np.zeros(len(data), dtype=int)
        position = 0
        for i in range(len(data)):
            if position != 0:
                # Only an opposite qualifying entry condition exits a position.
                if raw[i] == -position:
                    position = 0
            elif confirmed[i] != 0:
                position = int(confirmed[i])
            positions[i] = position
        df = pd.DataFrame({"signal": positions, "size": 1.0}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


if __name__ == "__main__":
    strategy = GeneratedStrategy()
    params = RepairParams()
    index = pd.date_range("2020-01-01", periods=9, freq="D")
    data = pd.DataFrame({"open": 100.0, "high": 101.0, "low": 99.0, "close": 100.0, "volume": 1000.0}, index=index)
    derived = pd.DataFrame({"entry_direction": [0, 1, 1, 0, -1, -1, -1, 0, 0]}, index=index)
    derived["confirmed_direction"] = derived["entry_direction"].where(
        derived["entry_direction"].eq(derived["entry_direction"].shift(1)), 0
    ).astype(int)
    signals = strategy.generate_signals(data, derived, None, params).data
    assert signals["signal"].tolist() == [0, 0, 0, 1, 1, 0, -1, -1, -1]
    assert signals["size"].gt(0).all()
    assert strategy.indicators(data, params)["entry_direction"].eq(0).all()
    assert strategy.warmup_bars(params) >= 13
