from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class VisibilityParams:
    topology_window: int = 12
    min_reach: float = 0.4
    min_gain: float = 0.25
    profit_target: float = 0.007
    max_hold_bars: int = 2


class GeneratedStrategy(BaseStrategy[VisibilityParams]):
    strategy_id = "gen_local_1791337693"

    @classmethod
    def params_type(cls) -> type[VisibilityParams]:
        return VisibilityParams

    def warmup_bars(self, params: VisibilityParams) -> int:
        if params.topology_window < 3:
            raise ValueError("topology_window must be at least 3")
        if not 0 < params.min_reach <= 1:
            raise ValueError("min_reach must be in (0, 1]")
        if not 0 < params.min_gain < 1:
            raise ValueError("min_gain must be in (0, 1)")
        if not np.isfinite(params.profit_target) or params.profit_target <= 0:
            raise ValueError("profit_target must be finite and positive")
        if params.max_hold_bars not in (1, 2):
            raise ValueError("max_hold_bars must be 1 or 2")
        return int(params.topology_window + 2)

    def indicators(self, data: pd.DataFrame, params: VisibilityParams) -> pd.DataFrame:
        self.warmup_bars(params)
        window = params.topology_window
        close = data["close"].astype(float).replace([np.inf, -np.inf], np.nan)
        close = close.where(close > 0)
        ready = close.rolling(window + 1, min_periods=window + 1).count().eq(window + 1)
        reach = pd.Series(1.0 / window, index=data.index)
        # Horizontal visibility: every interior close lies below both endpoints.
        # ponytail: O(window * bars); use a monotonic stack for very large windows.
        for lag in range(2, window + 1):
            interior_max = close.shift(1).rolling(lag - 1, min_periods=lag - 1).max()
            visible = close.gt(interior_max) & close.shift(lag).gt(interior_max)
            reach = reach.where(~visible, float(lag) / window)
        reach = reach.where(ready)
        gain = reach.diff()
        calendar_gate = pd.Series(
            (data.index.day >= 26) | (data.index.day <= 3), index=data.index
        )
        entry = (
            ready
            & ready.shift(1, fill_value=False)
            & calendar_gate
            & close.gt(close.shift(1))
            & reach.ge(params.min_reach)
            & gain.ge(params.min_gain)
        )
        strength = (gain / (2.0 * params.min_gain)).clip(0.0, 1.0) * reach
        return pd.DataFrame({
            "ready": ready,
            "calendar_gate": calendar_gate,
            "reach": reach.fillna(0.0),
            "reach_gain": gain.fillna(0.0),
            "strength": strength.fillna(0.0).clip(0.0, 1.0),
            "entry": entry.fillna(False),
        }, index=data.index)

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: VisibilityParams,
    ) -> SignalFrame:
        self.warmup_bars(params)
        count = len(data)
        signals = np.zeros(count, dtype=int)
        sizes = np.full(count, 0.2, dtype=float)
        entries = indicators["entry"].fillna(False).to_numpy(dtype=bool)
        strengths = indicators["strength"].fillna(0.0).to_numpy(dtype=float)
        opens = data["open"].to_numpy(dtype=float)
        closes = data["close"].to_numpy(dtype=float)
        active = False
        pending = False
        entry_price = 0.0
        entry_bar = -1
        held_size = 0.2
        for i in range(count):
            if pending:
                pending = False
                if np.isfinite(opens[i]) and opens[i] > 0:
                    entry_price = opens[i]
                    entry_bar = i
                    active = True
                else:
                    continue
            if active:
                held_bars = i - entry_bar + 1
                target_hit = (
                    np.isfinite(closes[i])
                    and closes[i] >= entry_price * (1.0 + params.profit_target)
                )
                # Close-based target decisions execute at the following open.
                if target_hit or held_bars >= params.max_hold_bars:
                    active = False
                    continue
                signals[i] = 1
                sizes[i] = held_size
            elif entries[i] and i + 1 < count:
                strength = strengths[i] if np.isfinite(strengths[i]) else 0.0
                held_size = float(0.2 + 0.75 * np.clip(strength, 0.0, 1.0))
                signals[i] = 1
                sizes[i] = held_size
                pending = True
        df = pd.DataFrame({"signal": signals, "size": sizes}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = df["size"].shift(1).fillna(0.2).astype(float)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check() -> None:
    strategy = GeneratedStrategy.__new__(GeneratedStrategy)
    params = VisibilityParams()
    index = pd.date_range("2024-01-26", periods=5, freq="B")
    data = pd.DataFrame({
        "open": 100.0,
        "high": 102.0,
        "low": 99.0,
        "close": 100.0,
        "volume": 1000.0,
    }, index=index)
    indicators = pd.DataFrame({
        "entry": [True, False, False, False, False],
        "strength": [0.8, 0.0, 0.0, 0.0, 0.0],
    }, index=index)
    result = strategy.generate_signals(data, indicators, None, params).data
    assert result["signal"].tolist() == [0, 1, 1, 0, 0]
    assert np.allclose(result.loc[result["signal"].eq(1), "size"], 0.8)
    data.loc[index[1], "close"] = 101.0
    result = strategy.generate_signals(data, indicators, None, params).data
    assert result["signal"].tolist() == [0, 1, 0, 0, 0]
    params.max_hold_bars = 1
    data["close"] = 100.0
    result = strategy.generate_signals(data, indicators, None, params).data
    assert result["signal"].tolist() == [0, 1, 0, 0, 0]
    derived = strategy.indicators(data, params)
    assert not derived.isna().any().any()
    assert not derived["entry"].any()
    assert result["size"].gt(0).all()


if __name__ == "__main__":
    _self_check()
