from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class OrdinalComplexityParams:
    window: int = 18
    complexity_floor: float = 0.10
    imbalance_floor: float = 0.20
    hold_bars: int = 2


class GeneratedStrategy(BaseStrategy[OrdinalComplexityParams]):
    strategy_id = "gen_local-w2_1791311317"

    @classmethod
    def params_type(cls):
        return OrdinalComplexityParams

    def warmup_bars(self, params: OrdinalComplexityParams) -> int:
        return int(params.window + 5)

    def indicators(self, data: pd.DataFrame, params: OrdinalComplexityParams) -> pd.DataFrame:
        if params.window < 6 or params.hold_bars not in (1, 2):
            raise ValueError("window must be >= 6 and hold_bars must be 1 or 2")
        if not 0 < params.complexity_floor < 1 or not 0 < params.imbalance_floor < 1:
            raise ValueError("complexity_floor and imbalance_floor must be between 0 and 1")
        close = data["close"].astype(float).replace([np.inf, -np.inf], np.nan)
        a, b, c = close.shift(2), close.shift(1), close
        conditions = [
            (a < b) & (b < c),
            (a < c) & (c < b),
            (b < a) & (a < c),
            (b < c) & (c < a),
            (c < a) & (a < b),
            (c < b) & (b < a),
        ]
        out = pd.DataFrame(index=data.index)
        out["pattern"] = np.select(conditions, [0, 1, 2, 3, 4, 5], default=np.nan)
        valid = out["pattern"].notna()
        probability_columns = []
        for k in range(6):
            column = f"probability_{k}"
            probability_columns.append(column)
            occurrence = out["pattern"].eq(k).astype(float).where(valid)
            out[column] = occurrence.rolling(params.window, min_periods=params.window).mean()
        probabilities = out[probability_columns]
        logs = np.log(probabilities.where(probabilities > 0, 1.0))
        out["entropy"] = -(probabilities * logs).sum(axis=1, min_count=6) / np.log(6.0)
        out["disequilibrium"] = ((probabilities - 1.0 / 6.0) ** 2).sum(axis=1, min_count=6) * 6.0 / 5.0
        out["complexity"] = out["entropy"] * out["disequilibrium"]
        out["imbalance"] = out["probability_0"] - out["probability_5"]
        out["imbalance_change"] = out["imbalance"].diff(3)
        eligible = (
            out["complexity"].ge(params.complexity_floor)
            & out["imbalance"].abs().ge(params.imbalance_floor)
            & (out["imbalance"] * out["imbalance_change"]).gt(0)
        )
        out["direction"] = np.sign(out["imbalance"]).where(eligible, 0).fillna(0).astype(int)
        out["conviction"] = (out["imbalance"].abs() * out["complexity"] / 0.15).clip(0, 1).fillna(0)
        return out

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: OrdinalComplexityParams) -> SignalFrame:
        if params.hold_bars not in (1, 2):
            raise ValueError("hold_bars must be 1 or 2")
        direction = indicators["direction"].fillna(0).to_numpy(dtype=int)
        conviction = indicators["conviction"].replace([np.inf, -np.inf], np.nan).fillna(0).clip(0, 1).to_numpy(dtype=float)
        signals = np.zeros(len(data), dtype=int)
        sizes = np.full(len(data), 0.20, dtype=float)
        active = 0
        exit_bar = -1
        entry_size = 0.20
        for i in range(len(data)):
            if active:
                if i == exit_bar:
                    active = 0
                    continue
            elif direction[i] != 0:
                active = int(direction[i])
                entry_size = float(0.20 + 0.75 * conviction[i])
                exit_bar = i + params.hold_bars
            if active:
                signals[i] = active
                sizes[i] = entry_size
        df = pd.DataFrame({"signal": signals, "size": sizes}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = df["size"].shift(1).fillna(0.20).astype(float)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


if __name__ == "__main__":
    index = pd.date_range("2020-01-01", periods=8, freq="D")
    data = pd.DataFrame({"open": 100.0, "high": 101.0, "low": 99.0, "close": 100.0, "volume": 1000.0}, index=index)
    indicators = pd.DataFrame({"direction": [1, -1, -1, -1, 1, 1, -1, 1], "conviction": [0.4, 0.9, 0.9, 0.8, 0.1, 0.1, 0.6, 0.1]}, index=index)
    strategy = GeneratedStrategy()
    result = strategy.generate_signals(data, indicators, None, OrdinalComplexityParams()).data
    assert result["signal"].tolist() == [0, 1, 1, 0, -1, -1, 0, -1]
    assert np.allclose(result["size"].iloc[1:3], 0.50)
    assert np.allclose(result["size"].iloc[4:6], 0.80)
    assert result["size"].gt(0).all()
