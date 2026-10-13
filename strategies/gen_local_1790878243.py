from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class RangeSheddingParams:
    range_window: int = 10
    expansion_ratio: float = 1.3
    contraction_ratio: float = 0.85
    migration_fraction: float = 0.2
    profit_target: float = 0.006
    hold_bars: int = 2


class GeneratedStrategy(BaseStrategy[RangeSheddingParams]):
    strategy_id = "gen_local_1790878243"

    @classmethod
    def params_type(cls) -> type[RangeSheddingParams]:
        return RangeSheddingParams

    @staticmethod
    def warmup_bars(params: RangeSheddingParams) -> int:
        return max(200, params.range_window + 2)

    def indicators(self, data: pd.DataFrame, params: RangeSheddingParams) -> pd.DataFrame:
        if params.range_window < 2 or params.expansion_ratio <= 0:
            raise ValueError("Invalid range window or expansion ratio")
        if not 0 < params.contraction_ratio < 1 or params.migration_fraction <= 0:
            raise ValueError("Invalid contraction ratio or migration fraction")
        out = pd.DataFrame(index=data.index)
        width = (data["high"] - data["low"]).where(lambda x: x > 0)
        previous_width = width.shift(1)
        out["range"] = width
        out["range_baseline"] = width.rolling(params.range_window, min_periods=params.range_window).mean().shift(1)
        out["ma200"] = data["close"].rolling(200, min_periods=200).mean()
        out["high_migration"] = data["high"].diff()
        out["low_migration"] = data["low"].diff()
        out["midpoint_migration"] = (out["high_migration"] + out["low_migration"]) / (2.0 * previous_width)
        out["contraction"] = width / previous_width
        out["previous_expansion"] = previous_width / out["range_baseline"].shift(1)
        qualified = (
            (out["previous_expansion"] >= params.expansion_ratio)
            & (out["contraction"] <= params.contraction_ratio)
        )
        long_entry = (
            qualified
            & (out["high_migration"] > 0)
            & (out["low_migration"] > 0)
            & (out["midpoint_migration"] >= params.migration_fraction)
            & (data["close"] > out["ma200"])
        )
        short_entry = (
            qualified
            & (out["high_migration"] < 0)
            & (out["low_migration"] < 0)
            & (out["midpoint_migration"] <= -params.migration_fraction)
            & (data["close"] < out["ma200"])
        )
        out["entry"] = np.select([long_entry.fillna(False), short_entry.fillna(False)], [1, -1], default=0).astype(int)
        return out

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: RangeSheddingParams) -> SignalFrame:
        if params.hold_bars not in (1, 2) or params.profit_target <= 0:
            raise ValueError("Use a positive profit target and a one- or two-bar holding period")
        entries = indicators["entry"].fillna(0).to_numpy(dtype=int)
        opens = data["open"].to_numpy(dtype=float)
        closes = data["close"].to_numpy(dtype=float)
        signals = np.zeros(len(data), dtype=int)
        position = 0
        entry_price = np.nan
        age = 0
        for i in range(len(data)):
            if position:
                # The previous close's decision fills at this bar's open.
                if age == 0:
                    entry_price = opens[i]
                age += 1
                valid_entry = np.isfinite(entry_price) and entry_price > 0
                gain = position * (closes[i] / entry_price - 1.0) if valid_entry and np.isfinite(closes[i]) else np.nan
                # Close-observed exits execute at the following open.
                if not valid_entry or gain >= params.profit_target or age >= params.hold_bars:
                    position = 0
                    entry_price = np.nan
                    age = 0
                signals[i] = position
            elif entries[i] and np.isfinite(closes[i]) and closes[i] > 0:
                position = int(entries[i])
                entry_price = np.nan
                age = 0
                signals[i] = position
        df = pd.DataFrame({"signal": signals, "size": 0.95}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check() -> None:
    strategy = GeneratedStrategy()
    params = RangeSheddingParams()
    index = pd.date_range("2020-01-01", periods=6)
    data = pd.DataFrame({"open": 100.0, "high": 102.0, "low": 98.0, "close": 100.0, "volume": 1000.0}, index=index)
    entries = pd.DataFrame({"entry": [1, 0, 0, 0, 0, 0]}, index=index)
    result = strategy.generate_signals(data, entries, None, params).data
    assert result["signal"].tolist() == [0, 1, 1, 0, 0, 0]
    data["close"] = [100.0, 101.0, 100.0, 100.0, 99.0, 100.0]
    entries["entry"] = [1, 0, 0, -1, 0, 0]
    result = strategy.generate_signals(data, entries, None, params).data
    assert result["signal"].tolist() == [0, 1, 0, 0, -1, 0]
    assert (result["size"] > 0).all()
    assert strategy.indicators(data, params)["entry"].eq(0).all()
    assert strategy.warmup_bars(params) >= max(200, params.range_window + 2)


if __name__ == "__main__":
    _self_check()
