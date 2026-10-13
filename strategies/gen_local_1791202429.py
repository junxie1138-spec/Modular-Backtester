from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class Params:
    frac_order: float = 0.45
    frac_window: int = 32
    strength_window: int = 10
    coherence_threshold: float = 0.65
    profit_target: float = 0.025
    max_hold_bars: int = 8


class GeneratedStrategy(BaseStrategy[Params]):
    strategy_id = "gen_local_1791202429"

    @classmethod
    def params_type(cls):
        return Params

    def warmup_bars(self, params: Params) -> int:
        return params.frac_window + params.strength_window + 1

    def indicators(self, data: pd.DataFrame, params: Params) -> pd.DataFrame:
        if not (0.0 < params.frac_order < 1.0):
            raise ValueError("frac_order must be between zero and one")
        if params.frac_window < 2 or params.strength_window < 3:
            raise ValueError("Indicator windows are too short")
        if not (0.0 < params.coherence_threshold < 1.0):
            raise ValueError("coherence_threshold must be between zero and one")
        if params.profit_target <= 0.0 or params.max_hold_bars < 1:
            raise ValueError("Exit parameters must be positive")
        close = data["close"].astype(float).where(data["close"] > 0.0)
        weights = np.ones(params.frac_window, dtype=float)
        for k in range(1, params.frac_window):
            weights[k] = -weights[k - 1] * (params.frac_order - k + 1) / k
        fractional_close = close * weights[0]
        for k in range(1, params.frac_window):
            fractional_close = fractional_close + weights[k] * close.shift(k)
        change = fractional_close.diff()
        total_movement = change.abs().rolling(params.strength_window).sum()
        coherence = change.rolling(params.strength_window).sum().div(
            total_movement.where(total_movement > 0.0)
        )
        recent_activity = change.abs().rolling(3).mean()
        baseline_activity = change.abs().rolling(params.strength_window).mean()
        expanding = recent_activity > baseline_activity
        candidate = pd.Series(0, index=data.index, dtype=int)
        candidate.loc[expanding & (coherence >= params.coherence_threshold)] = 1
        candidate.loc[expanding & (coherence <= -params.coherence_threshold)] = -1
        return pd.DataFrame({
            "fractional_close": fractional_close,
            "fractional_change": change,
            "total_movement": total_movement,
            "coherence": coherence,
            "recent_activity": recent_activity,
            "baseline_activity": baseline_activity,
            "expanding": expanding,
            "candidate": candidate,
        }, index=data.index)

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: Params,
    ) -> SignalFrame:
        candidate = indicators["candidate"].fillna(0).astype(int)
        confirmed = candidate.where(
            candidate.eq(candidate.shift(1)) & candidate.ne(0), 0
        ).to_numpy()
        opens = data["open"].to_numpy(dtype=float)
        closes = data["close"].to_numpy(dtype=float)
        desired = np.zeros(len(data), dtype=int)
        position = 0
        entry_bar = -1
        entry_price = np.nan
        for i in range(len(data)):
            if position:
                if i == entry_bar:
                    entry_price = opens[i]
                    if not np.isfinite(entry_price) or entry_price <= 0.0:
                        position = 0
                        continue
                gain = position * (closes[i] / entry_price - 1.0)
                if gain >= params.profit_target or i - entry_bar + 1 >= params.max_hold_bars:
                    position = 0
                    continue
            elif confirmed[i]:
                position = int(confirmed[i])
                entry_bar = i + 1
                entry_price = np.nan
            desired[i] = position
        df = pd.DataFrame({"signal": desired, "size": 0.95}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check() -> None:
    index = pd.date_range("2020-01-01", periods=8)
    data = pd.DataFrame({
        "open": 100.0, "high": 105.0, "low": 99.0,
        "close": [100.0, 100.0, 104.0, 100.0, 100.0, 100.0, 100.0, 100.0],
        "volume": 1000.0,
    }, index=index)
    indicators = pd.DataFrame({"candidate": [1, 1, 1, 0, -1, -1, -1, -1]}, index=index)
    strategy = GeneratedStrategy()
    params = Params(max_hold_bars=2)
    ctx = StrategyContext(symbol="SPY", timeframe="1d", warmup_bars=0)
    result = strategy.generate_signals(data, indicators, ctx, params).data
    assert result["signal"].tolist() == [0, 0, 1, 0, 0, 0, -1, -1]
    for end in range(1, len(data) + 1):
        prefix = strategy.generate_signals(data.iloc[:end], indicators.iloc[:end], ctx, params).data
        assert prefix.equals(result.iloc[:end])
    calculated = strategy.indicators(data, params)
    assert calculated["candidate"].eq(0).all()
    assert result["size"].gt(0).all()


if __name__ == "__main__":
    _self_check()
