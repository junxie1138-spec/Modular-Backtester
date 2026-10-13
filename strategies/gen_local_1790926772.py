from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class WeekendReleaseParams:
    acceleration_min: float = 0.001
    friday_volume_ratio: float = 1.1
    profit_target: float = 0.012
    hold_bars: int = 4


class GeneratedStrategy(BaseStrategy[WeekendReleaseParams]):
    strategy_id = "gen_local_1790926772"

    @classmethod
    def params_type(cls):
        return WeekendReleaseParams

    def warmup_bars(self, params):
        return 5

    def indicators(self, data, params):
        out = pd.DataFrame(index=data.index)
        out["roc"] = data["close"].pct_change(fill_method=None)
        out["acceleration"] = out["roc"].diff()
        baseline = data["volume"].shift(1).rolling(3, min_periods=3).mean()
        out["volume_ratio"] = data["volume"].div(baseline.where(baseline > 0))
        out["friday_roc"] = out["roc"].shift(1)
        out["friday_volume_ratio"] = out["volume_ratio"].shift(1)
        weekday = pd.Series(data.index.dayofweek, index=data.index)
        out["weekend_transition"] = weekday.eq(0) & weekday.shift(1).eq(4)
        return out.replace([np.inf, -np.inf], np.nan)

    def generate_signals(self, data, indicators, ctx, params):
        if not np.isfinite(params.acceleration_min) or params.acceleration_min < 0:
            raise ValueError("acceleration_min must be finite and nonnegative")
        if not np.isfinite(params.friday_volume_ratio) or params.friday_volume_ratio <= 0:
            raise ValueError("friday_volume_ratio must be finite and positive")
        if not np.isfinite(params.profit_target) or params.profit_target <= 0:
            raise ValueError("profit_target must be finite and positive")
        if params.hold_bars not in (3, 4, 5):
            raise ValueError("hold_bars must be 3, 4, or 5")
        entries = (
            indicators["weekend_transition"]
            & indicators["friday_roc"].lt(0)
            & indicators["friday_volume_ratio"].ge(params.friday_volume_ratio)
            & indicators["roc"].le(0)
            & indicators["acceleration"].ge(params.acceleration_min)
        ).fillna(False).to_numpy(dtype=bool)
        opens = data["open"].to_numpy(dtype=float)
        closes = data["close"].to_numpy(dtype=float)
        signal = np.zeros(len(data), dtype=int)
        entry_bar = -1
        entry_price = np.nan
        for i in range(len(data)):
            if entry_bar >= 0:
                if i == entry_bar:
                    entry_price = opens[i]
                invalid_entry = not np.isfinite(entry_price) or entry_price <= 0
                target_hit = np.isfinite(closes[i]) and closes[i] >= entry_price * (1 + params.profit_target)
                expired = i - entry_bar + 1 >= params.hold_bars
                if invalid_entry or target_hit or expired:
                    entry_bar = -1
                    entry_price = np.nan
                else:
                    signal[i] = 1
            elif entries[i] and i + 1 < len(data):
                entry_bar = i + 1
                signal[i] = 1
        df = pd.DataFrame({"signal": signal, "size": 0.95}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


if __name__ == "__main__":
    index = pd.bdate_range("2024-01-01", periods=10)
    close = np.array([100, 100, 100, 100, 98, 97, 98, 99, 100, 101], dtype=float)
    opening = np.concatenate(([100.0], close[:-1]))
    data = pd.DataFrame({"open": opening, "high": np.maximum(opening, close), "low": np.minimum(opening, close), "close": close, "volume": [100, 100, 100, 100, 300, 100, 100, 100, 100, 100]}, index=index)
    strategy = GeneratedStrategy()
    params = WeekendReleaseParams()
    derived = strategy.indicators(data, params)
    result = strategy.generate_signals(data, derived, None, params).data
    assert result["signal"].tolist() == [0, 0, 0, 0, 0, 0, 1, 1, 0, 0]
    params = WeekendReleaseParams(profit_target=1.0, hold_bars=3)
    result = strategy.generate_signals(data, derived, None, params).data
    assert result["signal"].tolist() == [0, 0, 0, 0, 0, 0, 1, 1, 1, 0]
