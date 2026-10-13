from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class StreakParams:
    streak_days: int = 2
    hold_bars: int = 4
    spike_multiple: float = 2.5

    def __post_init__(self):
        if not 2 <= self.streak_days <= 4 or not 3 <= self.hold_bars <= 5:
            raise ValueError("streak_days must be 2-4; hold_bars must be 3-5")
        if not np.isfinite(self.spike_multiple) or self.spike_multiple <= 1:
            raise ValueError("spike_multiple must be finite and greater than 1")


class GeneratedStrategy(BaseStrategy[StreakParams]):
    strategy_id = "gen_local-w2_1791258518"

    @classmethod
    def params_type(cls):
        return StreakParams

    def warmup_bars(self, params: StreakParams) -> int:
        # Return difference + five-bar scale + one-bar lag + three-bar cooldown.
        return 9

    def indicators(self, data: pd.DataFrame, params: StreakParams) -> pd.DataFrame:
        out = pd.DataFrame(index=data.index)
        returns = data["close"].where(data["close"] > 0).pct_change(fill_method=None)
        out["return"] = returns.replace([np.inf, -np.inf], np.nan)
        out["direction"] = np.sign(out["return"]).fillna(0).astype(int)
        active = out["direction"].ne(0)
        streak = active.astype(int)
        for lag in range(1, 4):
            active = active & out["direction"].eq(out["direction"].shift(lag))
            streak = streak + active.astype(int)
        out["streak"] = streak
        valid_pair = out["return"].notna() & out["return"].shift(1).notna()
        out["alternation"] = (
            (out["direction"] * out["direction"].shift(1) < 0)
            .astype(float).where(valid_pair)
        )
        out["alternation_rate"] = out["alternation"].rolling(3, min_periods=3).mean()
        out["prior_alternation"] = out["alternation_rate"].shift(params.streak_days - 1)
        out["past_scale"] = out["return"].abs().rolling(5, min_periods=5).mean().shift(1)
        scale_ready = out["past_scale"].notna() & out["return"].notna()
        out["spike"] = (
            out["return"].abs() > params.spike_multiple * out["past_scale"]
        ).astype(float).where(scale_ready)
        out["recent_spike"] = out["spike"].rolling(3, min_periods=3).max()
        eligible = (
            out["streak"].eq(params.streak_days)
            & out["prior_alternation"].eq(1.0)
            & out["recent_spike"].eq(0.0)
            & out["past_scale"].gt(0)
        )
        out["entry"] = out["direction"].where(eligible, 0).astype(int)
        return out

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: StreakParams,
    ) -> SignalFrame:
        entries = indicators["entry"].fillna(0).to_numpy(dtype=int)
        signals = np.zeros(len(data), dtype=int)
        position = 0
        exit_bar = -1
        for i in range(self.warmup_bars(params), len(data)):
            if position and i == exit_bar:
                position = 0
                # A mandatory flat bar prevents an entry from cancelling the exit.
            elif position == 0 and entries[i] != 0:
                position = int(entries[i])
                exit_bar = i + params.hold_bars
            signals[i] = position
        df = pd.DataFrame({"signal": signals, "size": 1.0}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


if __name__ == "__main__":
    strategy, params = GeneratedStrategy(), StreakParams()
    index = pd.date_range("2020-01-01", periods=24)
    returns = np.tile([0.002, -0.002, 0.002, -0.002], 6)
    returns[9:12] = 0.002
    close = 100 * np.cumprod(1 + returns)
    data = pd.DataFrame(
        {"open": close, "high": close * 1.01, "low": close * 0.99,
         "close": close, "volume": 1000.0}, index=index,
    )
    derived = strategy.indicators(data, params)
    assert derived["entry"].iloc[9] == 1
    assert derived.iloc[:16].equals(strategy.indicators(data.iloc[:16], params))
    derived["entry"] = 0
    derived.iloc[[9, 10, 13, 14], derived.columns.get_loc("entry")] = [1, -1, -1, -1]
    ctx = StrategyContext(symbol="SPY", timeframe="1d", warmup_bars=9)
    result = strategy.generate_signals(data, derived, ctx, params).data
    expected = np.zeros(24, dtype=int)
    expected[10:14], expected[15:19] = 1, -1
    assert np.array_equal(result["signal"].to_numpy(), expected)
    assert result["size"].gt(0).all()
