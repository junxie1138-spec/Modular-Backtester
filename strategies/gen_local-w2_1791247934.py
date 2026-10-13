from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class Params:
    fractional_order: float = 0.5
    profit_target: float = 0.012
    holding_bars: int = 4


class GeneratedStrategy(BaseStrategy[Params]):
    strategy_id = "gen_local-w2_1791247934"

    @classmethod
    def params_type(cls):
        return Params

    def warmup_bars(self, params: Params) -> int:
        return 9

    def indicators(self, data: pd.DataFrame, params: Params) -> pd.DataFrame:
        if not 0.0 < params.fractional_order < 1.0:
            raise ValueError("fractional_order must be between zero and one")
        if not np.isfinite(params.profit_target) or params.profit_target <= 0:
            raise ValueError("profit_target must be positive and finite")
        if params.holding_bars not in (3, 4, 5):
            raise ValueError("holding_bars must be 3, 4, or 5")
        close = data["close"].astype(float).where(data["close"] > 0)
        d = params.fractional_order
        weights = [1.0, -d, d * (d - 1.0) / 2.0,
                   -d * (d - 1.0) * (d - 2.0) / 6.0]
        # ponytail: four-tap fractional filter; extend memory if warmup limits relax.
        fractional_close = sum(w * close.shift(k) for k, w in enumerate(weights))
        anchor = close.shift(3)
        pressure = (fractional_close - sum(weights) * anchor) / anchor
        out = pd.DataFrame(index=data.index)
        out["fractional_close"] = fractional_close
        out["pressure"] = pressure
        out["previous_pressure"] = pressure.shift(1)
        out["weekly_pressure"] = pressure.shift(5)
        weekday = pd.Series(data.index.dayofweek, index=data.index)
        out["weekday"] = weekday
        out["previous_weekday"] = weekday.shift(5)
        out["entry"] = (
            (out["weekday"] == out["previous_weekday"])
            & (out["weekly_pressure"] < 0)
            & (out["pressure"] < 0)
            & (out["pressure"] > out["weekly_pressure"])
            & (out["pressure"] > out["previous_pressure"])
        ).fillna(False)
        return out

    def generate_signals(
        self, data: pd.DataFrame, indicators: pd.DataFrame,
        ctx: StrategyContext, params: Params,
    ) -> SignalFrame:
        opens = data["open"].to_numpy(dtype=float)
        closes = data["close"].to_numpy(dtype=float)
        entries = indicators["entry"].fillna(False).to_numpy(dtype=bool)
        signal = np.zeros(len(data), dtype=int)
        entry_bar = -1
        entry_price = np.nan
        for i in range(len(data)):
            if entry_bar >= 0:
                if i == entry_bar:
                    entry_price = opens[i]
                expired = i - entry_bar + 1 >= params.holding_bars
                invalid_fill = not np.isfinite(entry_price) or entry_price <= 0
                target = (
                    np.isfinite(closes[i])
                    and closes[i] >= entry_price * (1.0 + params.profit_target)
                )
                if expired or invalid_fill or target:
                    entry_bar = -1
                    entry_price = np.nan
                else:
                    signal[i] = 1
            elif entries[i]:
                signal[i] = 1
                entry_bar = i + 1
        df = pd.DataFrame({"signal": signal, "size": 0.95}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


if __name__ == "__main__":
    strategy = GeneratedStrategy()
    params = Params()
    index = pd.bdate_range("2024-01-01", periods=12)
    data = pd.DataFrame(
        {"open": 100.0, "high": 102.0, "low": 99.0,
         "close": 100.0, "volume": 1000000.0}, index=index,
    )
    ctx = StrategyContext(symbol="SPY", timeframe="1d", warmup_bars=9)
    entries = pd.DataFrame({"entry": [True] + [False] * 11}, index=index)
    timed = strategy.generate_signals(data, entries, ctx, params).data
    assert timed["signal"].tolist() == [0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0]
    data.loc[index[2], "close"] = 102.0
    targeted = strategy.generate_signals(data, entries, ctx, params).data
    assert targeted["signal"].tolist() == [0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    derived = strategy.indicators(data, params)
    assert not derived["entry"].iloc[:8].any()
    for end in range(9, len(data) + 1):
        pd.testing.assert_frame_equal(
            derived.iloc[:end], strategy.indicators(data.iloc[:end], params)
        )
