from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class Params:
    pattern_window: int = 8
    range_window: int = 10
    min_entropy: float = 0.80
    min_patterns: int = 5
    edge_fraction: float = 0.20
    hold_bars: int = 2


class GeneratedStrategy(BaseStrategy[Params]):
    strategy_id = "gen_local_1791290125"

    @classmethod
    def params_type(cls) -> type[Params]:
        return Params

    def warmup_bars(self, params: Params) -> int:
        if not 3 <= params.pattern_window <= 8:
            raise ValueError("pattern_window must be between 3 and 8")
        if not 2 <= params.range_window <= 10:
            raise ValueError("range_window must be between 2 and 10")
        if not 0.0 <= params.min_entropy <= 1.0:
            raise ValueError("min_entropy must be between 0 and 1")
        if not 1 <= params.min_patterns <= 6:
            raise ValueError("min_patterns must be between 1 and 6")
        if not 0.0 < params.edge_fraction < 0.5:
            raise ValueError("edge_fraction must be between 0 and 0.5")
        if params.hold_bars not in (1, 2):
            raise ValueError("hold_bars must be 1 or 2")
        return int(max(params.pattern_window + 2, params.range_window))

    def indicators(self, data: pd.DataFrame, params: Params) -> pd.DataFrame:
        self.warmup_bars(params)
        out = pd.DataFrame(index=data.index)
        a, b, c = data["close"].shift(2), data["close"].shift(1), data["close"]
        valid = np.isfinite(a) & np.isfinite(b) & np.isfinite(c)
        out["pattern"] = np.select(
            [
                (a < b) & (b < c),
                (a < c) & (c < b),
                (b < a) & (a < c),
                (b < c) & (c < a),
                (c < a) & (a < b),
                (c < b) & (b < a),
            ],
            [0.0, 1.0, 2.0, 3.0, 4.0, 5.0],
            default=np.nan,
        )
        out["pattern"] = out["pattern"].where(valid)
        probability_columns = []
        for k in range(6):
            name = f"probability_{k}"
            probability_columns.append(name)
            out[name] = (
                out["pattern"].eq(k).astype(float).where(out["pattern"].notna())
                .rolling(params.pattern_window, min_periods=params.pattern_window).mean()
            )
        probabilities = out[probability_columns]
        out["entropy"] = (
            -(probabilities * np.log(probabilities.where(probabilities > 0, 1.0)))
            .sum(axis=1, min_count=6) / np.log(6.0)
        )
        out["patterns_present"] = probabilities.gt(0).sum(axis=1).where(
            probabilities.notna().all(axis=1)
        )
        out["range_low"] = data["low"].rolling(params.range_window).min()
        out["range_high"] = data["high"].rolling(params.range_window).max()
        out["range_width"] = out["range_high"] - out["range_low"]
        out["relative_position"] = (
            (c - out["range_low"]) / out["range_width"].where(out["range_width"] > 0)
        )
        return out.replace([np.inf, -np.inf], np.nan)

    def generate_signals(
        self, data: pd.DataFrame, indicators: pd.DataFrame,
        ctx: StrategyContext, params: Params,
    ) -> SignalFrame:
        self.warmup_bars(params)
        eligible = (
            indicators["entropy"].ge(params.min_entropy)
            & indicators["patterns_present"].ge(params.min_patterns)
            & indicators["relative_position"].notna()
        )
        entries = np.select(
            [
                eligible & indicators["relative_position"].le(params.edge_fraction),
                eligible & indicators["relative_position"].ge(1.0 - params.edge_fraction),
            ],
            [1, -1], default=0,
        )
        signals = np.zeros(len(data), dtype=int)
        position, age = 0, 0
        for i, entry in enumerate(entries):
            if position:
                age += 1
                if age == params.hold_bars:
                    position, age = 0, 0
            elif entry:
                position, age = int(entry), 0
            signals[i] = position
        df = pd.DataFrame({"signal": signals, "size": 0.95}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


if __name__ == "__main__":
    strategy = GeneratedStrategy()
    params = Params()
    index = pd.date_range("2020-01-01", periods=12)
    close = pd.Series([100, 102, 101, 104, 99, 103, 98, 105, 100, 106, 101, 104], index=index)
    data = pd.DataFrame({
        "open": close, "high": close + 1, "low": close - 1,
        "close": close, "volume": 1000.0,
    })
    derived = strategy.indicators(data, params)
    assert strategy.warmup_bars(params) == 10
    assert derived["entropy"].iloc[:9].isna().all()
    assert derived["entropy"].iloc[9:].between(0, 1).all()
    test_data = data.iloc[:6]
    forced = pd.DataFrame({
        "entropy": 1.0, "patterns_present": 6,
        "relative_position": [0.0, 1.0, 1.0, 1.0, 0.0, 0.0],
    }, index=test_data.index)
    for hold, expected in [
        (2, [0, 1, 1, 0, -1, -1]),
        (1, [0, 1, 0, -1, 0, 1]),
    ]:
        result = strategy.generate_signals(test_data, forced, None, Params(hold_bars=hold)).data
        assert result["signal"].tolist() == expected
        assert result["size"].gt(0).all()
