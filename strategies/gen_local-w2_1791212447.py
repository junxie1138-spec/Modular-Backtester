from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class BreakoutCreditParams:
    lookback: int = 9
    efficiency_min: float = 0.4
    credit_cap: float = 4.0

    def __post_init__(self):
        if not 2 <= self.lookback <= 9:
            raise ValueError("lookback must be between 2 and 9")
        if not 0.0 <= self.efficiency_min <= 1.0:
            raise ValueError("efficiency_min must be between 0 and 1")
        if not np.isfinite(self.credit_cap) or self.credit_cap <= 0:
            raise ValueError("credit_cap must be positive and finite")


class GeneratedStrategy(BaseStrategy[BreakoutCreditParams]):
    strategy_id = "gen_local-w2_1791212447"

    @classmethod
    def params_type(cls):
        return BreakoutCreditParams

    def warmup_bars(self, params: BreakoutCreditParams) -> int:
        return params.lookback + 1

    def indicators(self, data: pd.DataFrame, params: BreakoutCreditParams) -> pd.DataFrame:
        out = pd.DataFrame(index=data.index)
        close = data["close"]
        out["change"] = close.diff()
        out["absolute_change"] = out["change"].abs()
        out["path"] = out["absolute_change"].rolling(params.lookback).sum()
        out["displacement"] = close.diff(params.lookback)
        out["close_rank"] = close.rolling(params.lookback).rank(method="min", pct=True)
        out["efficiency"] = out["displacement"].div(out["path"].replace(0.0, np.nan))
        valid = out["close_rank"].notna() & out["efficiency"].notna()
        breakout = valid & out["close_rank"].eq(1.0) & out["efficiency"].ge(params.efficiency_min)
        rejection = valid & out["close_rank"].lt(0.5)
        out["credit_change"] = np.select([breakout, rejection], [1.0, -0.5], default=0.0)
        return out

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: BreakoutCreditParams,
    ) -> SignalFrame:
        credit = 0.0
        signals = np.zeros(len(data), dtype=int)
        for i, change in enumerate(indicators["credit_change"].fillna(0.0).to_numpy()):
            credit = float(np.clip(credit + change, 0.0, params.credit_cap))
            # The entry predicate is positive breakout credit; exit only on its reversal.
            signals[i] = int(credit > 0.0)
        df = pd.DataFrame({"signal": signals, "size": 1.0}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


if __name__ == "__main__":
    params = BreakoutCreditParams()
    strategy = GeneratedStrategy()
    prices = np.concatenate([100.0 + np.arange(50), 149.0 - np.arange(1, 51)])
    sample = pd.DataFrame(
        {"open": prices, "high": prices + 1, "low": prices - 1,
         "close": prices, "volume": 1000000.0},
        index=pd.date_range("2020-01-01", periods=len(prices)),
    )
    ctx = StrategyContext(symbol="SPY", timeframe="1d", warmup_bars=strategy.warmup_bars(params))
    result = strategy.generate_signals(sample, strategy.indicators(sample, params), ctx, params).data
    assert strategy.warmup_bars(params) <= 10
    assert result["signal"].iloc[:params.lookback + 1].eq(0).all()
    assert result["signal"].iloc[params.lookback + 1] == 1
    assert result["signal"].iloc[49] == 1 and result["signal"].iloc[-1] == 0
    assert result["signal"].isin([0, 1]).all() and result["size"].gt(0).all()
    prefix = sample.iloc[:60]
    causal = strategy.generate_signals(prefix, strategy.indicators(prefix, params), ctx, params).data
    pd.testing.assert_frame_equal(causal, result.iloc[:60])
