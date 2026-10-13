from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class StreakParams:
    streak_window: int = 63
    regime_window: int = 20
    regime_history: int = 126
    streak_quantile: float = 0.85
    regime_quantile: float = 0.70
    profit_target: float = 0.008
    max_hold: int = 2


class GeneratedStrategy(BaseStrategy[StreakParams]):
    strategy_id = "gen_local-w2_1791277893"

    @classmethod
    def params_type(cls):
        return StreakParams

    def warmup_bars(self, params):
        return int(max(params.streak_window + 1,
                       params.regime_window + params.regime_history + 2))

    def indicators(self, data, params):
        if (min(params.streak_window, params.regime_window,
                params.regime_history) < 2
                or not 0.5 < params.streak_quantile < 1.0
                or not 0.5 < params.regime_quantile < 1.0
                or params.profit_target <= 0 or params.max_hold not in (1, 2)):
            raise ValueError("Invalid streak, percentile, target, or holding parameters")
        out = pd.DataFrame(index=data.index)
        direction = np.sign(data["close"].diff())
        groups = direction.fillna(0).ne(direction.shift(1).fillna(0)).cumsum()
        streak = direction.groupby(groups).cumcount().add(1)
        streak = streak.where(direction.notna() & direction.ne(0), 0)
        out["direction"] = direction
        out["streak"] = streak
        out["streak_threshold"] = streak.shift(1).rolling(
            params.streak_window, min_periods=params.streak_window
        ).quantile(params.streak_quantile).clip(lower=2)
        out["persistence"] = (direction * direction.shift(1)).rolling(
            params.regime_window, min_periods=params.regime_window
        ).mean()
        history = out["persistence"].shift(1).rolling(
            params.regime_history, min_periods=params.regime_history
        )
        out["regime_high"] = history.quantile(params.regime_quantile)
        out["regime_low"] = history.quantile(1.0 - params.regime_quantile)
        continuation = out["persistence"].gt(out["regime_high"])
        reversal = out["persistence"].lt(out["regime_low"])
        rare = streak.ge(np.ceil(out["streak_threshold"]))
        onset = rare & ~rare.shift(1, fill_value=False)
        valid = out[["direction", "streak_threshold", "persistence",
                     "regime_high", "regime_low"]].notna().all(axis=1)
        out["entry_direction"] = np.where(
            valid & onset & continuation, direction,
            np.where(valid & onset & reversal, -direction, 0)
        ).astype(int)
        return out

    def generate_signals(self, data, indicators, ctx, params):
        decisions = np.zeros(len(data), dtype=int)
        entries = indicators["entry_direction"].to_numpy()
        opens = data["open"].to_numpy(dtype=float)
        closes = data["close"].to_numpy(dtype=float)
        position = 0
        entry_price = np.nan
        age = 0
        for i in range(len(data)):
            if position:
                # The previous close's entry decision fills at this open.
                if np.isnan(entry_price):
                    entry_price = opens[i]
                if not np.isfinite(entry_price) or entry_price <= 0:
                    position = 0
                    entry_price = np.nan
                    continue
                age += 1
                gain = position * (closes[i] / entry_price - 1.0)
                if gain >= params.profit_target or age >= params.max_hold:
                    position = 0
                    entry_price = np.nan
                    age = 0
                # An exit decision remains flat until a later close.
            elif entries[i]:
                position = int(entries[i])
                entry_price = np.nan
                age = 0
            decisions[i] = position
        df = pd.DataFrame({"signal": decisions, "size": 1.0}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check():
    index = pd.date_range("2020-01-01", periods=8)
    data = pd.DataFrame({
        "open": 100.0, "high": 101.0, "low": 99.0,
        "close": [100., 100., 101., 100., 100., 100., 100., 100.],
        "volume": 1000.0,
    }, index=index)
    entries = pd.DataFrame(
        {"entry_direction": [1, 0, 0, 0, -1, 0, 0, 0]}, index=index
    )
    result = GeneratedStrategy().generate_signals(
        data, entries, None, StreakParams()
    ).data
    assert result["signal"].tolist() == [0, 1, 1, 0, 0, -1, -1, 0]
    assert result["size"].gt(0).all()


if __name__ == "__main__":
    _self_check()
