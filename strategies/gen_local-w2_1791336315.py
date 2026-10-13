from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class CycleParams:
    profit_target: float = 0.04
    holding_bars: int = 18


class GeneratedStrategy(BaseStrategy[CycleParams]):
    strategy_id = "gen_local-w2_1791336315"

    @classmethod
    def params_type(cls):
        return CycleParams

    @staticmethod
    def warmup_bars(params: CycleParams) -> int:
        # Five-session return, 21 ranks, ten-session lag, 63 correlations.
        return 98

    def indicators(self, data: pd.DataFrame, params: CycleParams) -> pd.DataFrame:
        close = data["close"].where(data["close"] > 0)
        out = pd.DataFrame(index=data.index)
        out["weekly_return"] = close.pct_change(5, fill_method=None).replace([np.inf, -np.inf], np.nan)
        out["daily_return"] = close.pct_change(fill_method=None).replace([np.inf, -np.inf], np.nan)
        out["percentile"] = out["weekly_return"].rolling(21, min_periods=21).rank(pct=True)
        out["previous_percentile"] = out["percentile"].shift(1)
        out["lag_percentile"] = out["percentile"].shift(10)
        out["cycle_correlation"] = out["percentile"].rolling(63, min_periods=63).corr(out["lag_percentile"]).replace([np.inf, -np.inf], np.nan)
        out["entry"] = (
            (out["percentile"] > 0.35)
            & (out["previous_percentile"] <= 0.35)
            & (out["lag_percentile"] >= 0.65)
            & (out["cycle_correlation"] < -0.20)
            & (out["daily_return"] > 0)
        ).fillna(False)
        return out

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: CycleParams,
    ) -> SignalFrame:
        if not np.isfinite(params.profit_target) or params.profit_target <= 0 or params.holding_bars < 1:
            raise ValueError("profit_target must be positive and holding_bars at least one")
        entries = indicators["entry"].fillna(False).to_numpy(dtype=bool)
        opens = data["open"].to_numpy(dtype=float)
        closes = data["close"].to_numpy(dtype=float)
        signals = np.zeros(len(data), dtype=int)
        active = False
        entry_bar = -1
        entry_price = np.nan
        for i in range(len(data)):
            if active:
                if i == entry_bar:
                    # Observe the fill bar's open only after that bar arrives.
                    entry_price = opens[i]
                if not np.isfinite(entry_price) or entry_price <= 0:
                    active = False
                else:
                    reached_target = np.isfinite(closes[i]) and closes[i] >= entry_price * (1.0 + params.profit_target)
                    expired = i - entry_bar + 1 >= params.holding_bars
                    if reached_target or expired:
                        active = False
            elif entries[i] and np.isfinite(closes[i]) and closes[i] > 0:
                active = True
                entry_bar = i + 1
                entry_price = np.nan
            signals[i] = int(active)
        df = pd.DataFrame({"signal": signals, "size": 0.95}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


if __name__ == "__main__":
    strategy = object.__new__(GeneratedStrategy)
    index = pd.date_range("2020-01-01", periods=6)
    bars = pd.DataFrame({"open": 100.0, "high": 106.0, "low": 99.0, "close": [100.0, 100.0, 105.0, 105.0, 105.0, 105.0], "volume": 1000.0}, index=index)
    entries = pd.DataFrame({"entry": [True, False, False, False, False, False]}, index=index)
    target = strategy.generate_signals(bars, entries, None, CycleParams()).data
    timed = strategy.generate_signals(bars, entries, None, CycleParams(profit_target=1.0, holding_bars=3)).data
    assert target["signal"].tolist() == [0, 1, 1, 0, 0, 0]
    assert timed["signal"].tolist() == [0, 1, 1, 1, 0, 0]
    assert not strategy.indicators(bars, CycleParams())["entry"].any()
