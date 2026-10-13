from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class GapPhaseParams:
    window: int = 40
    profit_target: float = 0.006


class GeneratedStrategy(BaseStrategy[GapPhaseParams]):
    strategy_id = "gen_local_1791327420"

    @classmethod
    def params_type(cls):
        return GapPhaseParams

    def warmup_bars(self, params: GapPhaseParams) -> int:
        if params.window < 3 or not np.isfinite(params.profit_target) or params.profit_target <= 0:
            raise ValueError("window must be >= 3 and profit_target must be finite and positive")
        return int(params.window) + 3

    def indicators(self, data: pd.DataFrame, params: GapPhaseParams) -> pd.DataFrame:
        self.warmup_bars(params)
        df = pd.DataFrame(index=data.index)
        previous_close = data["close"].shift(1).where(lambda x: x > 0)
        df["gap"] = (data["open"] / previous_close - 1).replace([np.inf, -np.inf], np.nan)
        df["gap_lag1"] = df["gap"].shift(1)
        df["gap_lag2"] = df["gap"].shift(2)
        df["corr1"] = df["gap"].rolling(params.window).corr(df["gap_lag1"]).replace([np.inf, -np.inf], np.nan).fillna(0.0)
        df["corr2"] = df["gap"].rolling(params.window).corr(df["gap_lag2"]).replace([np.inf, -np.inf], np.nan).fillna(0.0)
        df["entry"] = (
            (df["corr1"] < -0.15)
            & (df["corr2"] > 0.15)
            & (df["gap"] > 0)
            & (df["gap_lag1"] < 0)
            & (data["close"] > 0)
        ).fillna(False)
        return df

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: GapPhaseParams,
    ) -> SignalFrame:
        warmup = self.warmup_bars(params)
        entries = indicators["entry"].fillna(False).to_numpy(dtype=bool)
        opens = data["open"].to_numpy(dtype=float)
        closes = data["close"].to_numpy(dtype=float)
        decisions = np.zeros(len(data), dtype=int)
        active = False
        entry_bar = -1
        entry_price = 0.0
        for i in range(len(data)):
            if active:
                # Record the next-open entry only when that bar has arrived.
                if entry_bar == -1:
                    entry_bar = i
                    entry_price = opens[i]
                valid_entry = np.isfinite(entry_price) and entry_price > 0
                target_hit = valid_entry and np.isfinite(closes[i]) and closes[i] >= entry_price * (1 + params.profit_target)
                # Close-observed exits execute at the following open.
                if not valid_entry or target_hit or i - entry_bar + 1 >= 2:
                    active = False
            elif i >= warmup - 1 and entries[i]:
                active = True
                entry_bar = -1
            decisions[i] = int(active)
        df = pd.DataFrame({"signal": decisions, "size": 1.0}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check() -> None:
    strategy = GeneratedStrategy()
    params = GapPhaseParams(window=3)
    index = pd.date_range("2020-01-01", periods=11)
    data = pd.DataFrame(
        {"open": 100.0, "high": 102.0, "low": 99.0, "close": 100.0, "volume": 1000.0},
        index=index,
    )
    indicators = pd.DataFrame({"entry": [False] * 5 + [True] + [False] * 5}, index=index)
    ctx = StrategyContext(symbol="SPY", timeframe="1d", warmup_bars=6)
    timed = strategy.generate_signals(data, indicators, ctx, params).data
    assert timed["signal"].tolist() == [0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0]
    data.loc[index[6], "close"] = 101.0
    targeted = strategy.generate_signals(data, indicators, ctx, params).data
    assert targeted["signal"].tolist() == [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0]
    assert (targeted["size"] > 0).all()
    assert not strategy.indicators(data.iloc[:6], params)["entry"].any()


if __name__ == "__main__":
    _self_check()
