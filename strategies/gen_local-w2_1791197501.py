from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class RecoveryParams:
    window: int = 18
    downside_ratio: float = 0.8


class GeneratedStrategy(BaseStrategy[RecoveryParams]):
    strategy_id = "gen_local-w2_1791197501"

    @classmethod
    def params_type(cls) -> type[RecoveryParams]:
        return RecoveryParams

    def warmup_bars(self, params: RecoveryParams) -> int:
        return max(63, params.window + 1)

    def indicators(self, data: pd.DataFrame, params: RecoveryParams) -> pd.DataFrame:
        if params.window < 2 or not np.isfinite(params.downside_ratio) or params.downside_ratio <= 0:
            raise ValueError("window must be >= 2 and downside_ratio must be finite and positive")
        out = pd.DataFrame(index=data.index)
        out["price"] = data["close"].where(np.isfinite(data["close"]) & data["close"].gt(0))
        out["return"] = out["price"].pct_change(fill_method=None).replace([np.inf, -np.inf], np.nan)
        out["up_squared"] = out["return"].clip(lower=0).pow(2)
        out["down_squared"] = out["return"].clip(upper=0).pow(2)
        out["up_vol"] = np.sqrt(out["up_squared"].rolling(params.window, min_periods=params.window).mean())
        out["down_vol"] = np.sqrt(out["down_squared"].rolling(params.window, min_periods=params.window).mean())
        out["peak"] = out["price"].rolling(63, min_periods=63).max()
        out["drawdown"] = out["price"].div(out["peak"]).sub(1)
        out["recovery_return"] = out["price"].div(out["price"].shift(params.window)).sub(1)
        return out.replace([np.inf, -np.inf], np.nan)

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: RecoveryParams) -> SignalFrame:
        ready = indicators[["up_vol", "down_vol", "drawdown", "recovery_return"]].notna().all(axis=1)
        recovery = (
            ready
            & indicators["drawdown"].lt(0)
            & indicators["recovery_return"].gt(0)
            & indicators["up_vol"].gt(0)
            & indicators["down_vol"].lt(params.downside_ratio * indicators["up_vol"])
        )
        df = pd.DataFrame(index=data.index)
        # The same condition controls entry and exit; a reversal sets the target flat.
        df["signal"] = recovery.fillna(False).astype(int)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = 1.0
        return SignalFrame(data=df, signal_column="signal", size_column="size")


if __name__ == "__main__":
    prices = np.concatenate([np.full(63, 100.0), np.linspace(99.0, 85.0, 10), np.linspace(85.4, 99.0, 40), np.linspace(98.0, 80.0, 25)])
    data = pd.DataFrame({"open": prices, "high": prices + 0.5, "low": prices - 0.5, "close": prices, "volume": 1000000.0}, index=pd.date_range("2020-01-01", periods=len(prices)))
    strategy = GeneratedStrategy()
    params = RecoveryParams()
    derived = strategy.indicators(data, params)
    result = strategy.generate_signals(data, derived, None, params).data
    first_entry = np.flatnonzero(result["signal"].to_numpy() == 1)[0]
    assert derived["recovery_return"].iloc[first_entry - 1] > 0
    assert derived["down_vol"].iloc[first_entry - 1] < params.downside_ratio * derived["up_vol"].iloc[first_entry - 1]
    assert result["signal"].iloc[:63].eq(0).all()
    assert result["signal"].iloc[-1] == 0
    assert set(result["signal"].unique()) <= {0, 1}
    assert result["size"].gt(0).all()
    rising = data.copy()
    rising["close"] = np.arange(len(data), dtype=float) + 100.0
    assert strategy.generate_signals(rising, strategy.indicators(rising, params), None, params).data["signal"].eq(0).all()
