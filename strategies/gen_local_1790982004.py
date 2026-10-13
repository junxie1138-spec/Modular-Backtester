from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class Params:
    max_branching: float = 0.0
    profit_target: float = 0.012
    holding_bars: int = 4


class GeneratedStrategy(BaseStrategy[Params]):
    strategy_id = "gen_local_1790982004"

    @classmethod
    def params_type(cls):
        return Params

    def warmup_bars(self, params: Params) -> int:
        return 9

    def indicators(self, data: pd.DataFrame, params: Params) -> pd.DataFrame:
        if not 0.0 <= params.max_branching <= 1.0:
            raise ValueError("max_branching must be in [0, 1]")
        if not np.isfinite(params.profit_target) or params.profit_target <= 0:
            raise ValueError("profit_target must be positive and finite")
        if params.holding_bars not in (3, 4, 5):
            raise ValueError("holding_bars must be 3, 4, or 5")
        close = data["close"]
        a, b, c = close.shift(2), close.shift(1), close
        valid = np.isfinite(a) & np.isfinite(b) & np.isfinite(c)
        valid &= (a != b) & (a != c) & (b != c)
        pattern = (4 * (a < b).astype(int) + 2 * (a < c).astype(int)
                   + (b < c).astype(int)).astype(float).where(valid)
        complete = pattern.notna().rolling(6, min_periods=6).sum().eq(6)
        diversity = pd.Series(0.0, index=data.index)
        for code in (0, 1, 3, 4, 6, 7):
            count = pattern.eq(code).astype(float).rolling(6, min_periods=6).sum()
            diversity += count.gt(0).astype(float)
        repeats = pd.Series(0.0, index=data.index)
        branches = pd.Series(0.0, index=data.index)
        for i in range(5):
            for j in range(i + 1, 5):
                same = pattern.shift(i + 1).eq(pattern.shift(j + 1))
                repeats += same.astype(float)
                branches += (same & pattern.shift(i).ne(pattern.shift(j))).astype(float)
        branching = (branches / repeats.replace(0.0, np.nan)).where(complete)
        out = pd.DataFrame(index=data.index)
        out["pattern"] = pattern
        out["diversity"] = diversity.where(complete)
        out["repeated_antecedents"] = repeats.where(complete)
        out["branching"] = branching
        out["prior_diversity"] = out["diversity"].shift(1)
        out["prior_branching"] = branching.shift(1)
        out["channel_high"] = data["high"].rolling(6, min_periods=6).max().shift(1)
        out["entry"] = (close.gt(out["channel_high"])
                        & out["prior_diversity"].ge(3)
                        & out["prior_branching"].le(params.max_branching)).fillna(False)
        return out

    def generate_signals(
        self, data: pd.DataFrame, indicators: pd.DataFrame,
        ctx: StrategyContext, params: Params,
    ) -> SignalFrame:
        decisions = np.zeros(len(data), dtype=int)
        opens = data["open"].to_numpy(dtype=float)
        closes = data["close"].to_numpy(dtype=float)
        entries = indicators["entry"].fillna(False).to_numpy(dtype=bool)
        pending = False
        entry_bar = -1
        entry_price = np.nan
        for i in range(len(data)):
            if pending:
                pending = False
                if np.isfinite(opens[i]) and opens[i] > 0:
                    entry_bar, entry_price = i, opens[i]
                else:
                    continue
            if entry_bar >= 0:
                target_hit = np.isfinite(closes[i]) and closes[i] >= entry_price * (1 + params.profit_target)
                expired = i - entry_bar + 1 >= params.holding_bars
                if target_hit or expired:
                    entry_bar = -1
                else:
                    decisions[i] = 1
            elif entries[i]:
                decisions[i] = 1
                pending = True
        df = pd.DataFrame({"signal": decisions, "size": 0.95}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check():
    strategy, params = GeneratedStrategy(), Params()
    index = pd.date_range("2020-01-01", periods=12)
    data = pd.DataFrame(
        {"open": 100.0, "high": 101.0, "low": 99.0, "close": 100.0, "volume": 1000.0},
        index=index,
    )
    entries = pd.DataFrame({"entry": [True] + [False] * 11}, index=index)
    ctx = StrategyContext(symbol="SPY", timeframe="1d", warmup_bars=9)
    result = strategy.generate_signals(data, entries, ctx, params).data
    assert result["signal"].tolist() == [0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0]
    data.loc[index[2], "close"] = 102.0
    result = strategy.generate_signals(data, entries, ctx, params).data
    assert result["signal"].tolist() == [0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    prices = np.array([100, 102, 101, 100, 102, 101, 100, 102, 101, 104, 104, 104], dtype=float)
    data["open"] = data["close"] = prices
    data["high"], data["low"] = prices + 0.2, prices - 0.2
    indicators = strategy.indicators(data, params)
    assert indicators["entry"].iloc[9]
    assert not indicators["entry"].iloc[:8].any()
    for end in range(1, len(data) + 1):
        pd.testing.assert_frame_equal(
            strategy.indicators(data.iloc[:end], params), indicators.iloc[:end]
        )


if __name__ == "__main__":
    _self_check()
