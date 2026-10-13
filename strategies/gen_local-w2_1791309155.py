from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class ShelfParams:
    entry_z: float = 1.25
    profit_target: float = 0.025


class GeneratedStrategy(BaseStrategy[ShelfParams]):
    strategy_id = "gen_local-w2_1791309155"

    @classmethod
    def params_type(cls) -> type[ShelfParams]:
        return ShelfParams

    @staticmethod
    def warmup_bars(params: ShelfParams) -> int:
        return 25

    @staticmethod
    def indicators(data: pd.DataFrame, params: ShelfParams) -> pd.DataFrame:
        if not 0.35 < params.entry_z < 2.5:
            raise ValueError("entry_z must be between 0.35 and 2.5")
        if not np.isfinite(params.profit_target) or params.profit_target <= 0:
            raise ValueError("profit_target must be finite and positive")
        out = pd.DataFrame(index=data.index)
        close = data["close"]
        out["ma"] = close.rolling(20, min_periods=20).mean()
        out["sd"] = close.rolling(20, min_periods=20).std(ddof=0)
        out["z"] = ((close - out["ma"]) / out["sd"].replace(0.0, np.nan)).replace([np.inf, -np.inf], np.nan)
        out["shelf"] = out["z"].between(0.35, params.entry_z)
        out["prior_shelf_count"] = out["shelf"].shift(1).rolling(3, min_periods=3).sum()
        out["spike"] = out["z"].ge(2.5)
        out["prior_spike_count"] = out["spike"].shift(1).rolling(5, min_periods=5).sum()
        out["ma_change"] = out["ma"].diff()
        out["entry"] = (
            out["z"].gt(params.entry_z)
            & out["z"].lt(2.5)
            & out["prior_shelf_count"].eq(3)
            & out["prior_spike_count"].eq(0)
            & out["ma_change"].gt(0)
        ).fillna(False)
        return out

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame,
                         ctx: StrategyContext, params: ShelfParams) -> SignalFrame:
        signals = np.zeros(len(data), dtype=int)
        entries = indicators["entry"].fillna(False).to_numpy(dtype=bool)
        opens = data["open"].to_numpy(dtype=float)
        closes = data["close"].to_numpy(dtype=float)
        active = False
        pending_entry = False
        entry_price = 0.0
        entry_bar = -1
        for i in range(len(data)):
            if pending_entry:
                pending_entry = False
                if np.isfinite(opens[i]) and opens[i] > 0:
                    entry_price = opens[i]
                    entry_bar = i
                    active = True
            if active:
                # Targets use observed closes and nominal entry opens; fills occur next bar.
                target_hit = np.isfinite(closes[i]) and closes[i] >= entry_price * (1.0 + params.profit_target)
                time_up = i - entry_bar + 1 >= 8
                if target_hit or time_up:
                    active = False
                else:
                    signals[i] = 1
            elif entries[i] and i >= self.warmup_bars(params) - 1:
                signals[i] = 1
                pending_entry = True
        df = pd.DataFrame({"signal": signals, "size": 0.95}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


if __name__ == "__main__":
    params = ShelfParams()
    strategy = GeneratedStrategy()
    index = pd.date_range("2020-01-01", periods=40, freq="B")
    bars = pd.DataFrame({"open": 100.0, "high": 101.0, "low": 99.0,
                         "close": 100.0, "volume": 1000.0}, index=index)
    derived = strategy.indicators(bars, params)
    assert not derived["entry"].any()
    derived["entry"] = False
    derived.loc[index[25], "entry"] = True
    timed = strategy.generate_signals(bars, derived, None, params).data
    expected = np.zeros(40, dtype=int)
    expected[26:34] = 1
    assert np.array_equal(timed["signal"].to_numpy(), expected)
    bars.loc[index[27], "close"] = 103.0
    targeted = strategy.generate_signals(bars, derived, None, params).data
    expected[28:34] = 0
    assert np.array_equal(targeted["signal"].to_numpy(), expected)
    assert targeted["size"].gt(0).all()
