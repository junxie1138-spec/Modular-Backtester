from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class StrategyParams:
    streak_days: int = 3
    pressure_window: int = 6
    pressure_threshold: float = 0.15
    hold_bars: int = 4


class GeneratedStrategy(BaseStrategy[StrategyParams]):
    strategy_id = "gen_local_1791280704"

    @classmethod
    def params_type(cls) -> type[StrategyParams]:
        return StrategyParams

    def warmup_bars(self, params: StrategyParams) -> int:
        if params.streak_days < 1 or params.pressure_window < 1:
            raise ValueError("Lookbacks must be positive")
        if params.hold_bars < 3 or params.hold_bars > 5:
            raise ValueError("hold_bars must be between 3 and 5")
        if not np.isfinite(params.pressure_threshold) or not 0 <= params.pressure_threshold < 1:
            raise ValueError("pressure_threshold must be finite and in [0, 1)")
        return int(max(params.streak_days + 1, params.pressure_window))

    def indicators(self, data: pd.DataFrame, params: StrategyParams) -> pd.DataFrame:
        self.warmup_bars(params)
        out = pd.DataFrame(index=data.index)
        out["low_change"] = data["low"].diff()
        out["rising_low"] = out["low_change"].gt(0)
        up = out["rising_low"]
        out["low_streak"] = up.astype(int).groupby((~up).cumsum()).cumsum().clip(upper=params.streak_days)
        out["body_volume"] = (data["close"] - data["open"]) * data["volume"]
        out["range_volume"] = (data["high"] - data["low"]) * data["volume"]
        out["body_sum"] = out["body_volume"].rolling(params.pressure_window).sum()
        out["range_sum"] = out["range_volume"].rolling(params.pressure_window).sum()
        out["pressure"] = (out["body_sum"] / out["range_sum"].where(out["range_sum"] > 0)).replace([np.inf, -np.inf], np.nan)
        return out

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: StrategyParams) -> SignalFrame:
        warmup = self.warmup_bars(params)
        eligible = (
            indicators["low_streak"].ge(params.streak_days)
            & indicators["pressure"].gt(params.pressure_threshold)
        ).fillna(False).to_numpy(dtype=bool)
        eligible[:warmup - 1] = False
        signals = np.zeros(len(data), dtype=int)
        entry_bar = -1
        for i in range(len(data)):
            if entry_bar >= 0:
                if i - entry_bar >= params.hold_bars:
                    entry_bar = -1
                else:
                    signals[i] = 1
            elif eligible[i]:
                entry_bar = i
                signals[i] = 1
        df = pd.DataFrame({"signal": signals, "size": 1.0}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


if __name__ == "__main__":
    params = StrategyParams()
    strategy = GeneratedStrategy()
    index = pd.date_range("2020-01-01", periods=20, freq="D")
    low = 10.0 + np.arange(len(index))
    data = pd.DataFrame({"open": low + 0.5, "high": low + 2.0, "low": low, "close": low + 1.5, "volume": 100.0}, index=index)
    indicators = strategy.indicators(data, params)
    decision_bar = strategy.warmup_bars(params) - 1
    indicators.loc[index[decision_bar + 1:], "pressure"] = -1.0
    result = strategy.generate_signals(data, indicators, None, params).data
    expected = np.zeros(len(data), dtype=int)
    expected[decision_bar + 1:decision_bar + 1 + params.hold_bars] = 1
    assert np.array_equal(result["signal"].to_numpy(), expected)
    assert result["size"].gt(0).all()
    for column in ("pressure", "low_streak"):
        blocked = indicators.copy()
        blocked[column] = 0.0
        assert strategy.generate_signals(data, blocked, None, params).data["signal"].eq(0).all()
    zero_volume = data.copy()
    zero_volume["volume"] = 0.0
    assert strategy.generate_signals(zero_volume, strategy.indicators(zero_volume, params), None, params).data["signal"].eq(0).all()
