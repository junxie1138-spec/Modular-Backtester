from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class ShockwaveParams:
    min_streak: int = 3
    contraction: float = 0.8
    profit_target: float = 0.025
    hold_bars: int = 8


class GeneratedStrategy(BaseStrategy[ShockwaveParams]):
    strategy_id = "gen_local_1790934657"

    @classmethod
    def params_type(cls) -> type[ShockwaveParams]:
        return ShockwaveParams

    def warmup_bars(self, params: ShockwaveParams) -> int:
        return 8

    def indicators(self, data: pd.DataFrame, params: ShockwaveParams) -> pd.DataFrame:
        if not 3 <= params.min_streak <= 7:
            raise ValueError("min_streak must be between 3 and 7")
        if not 0.0 < params.contraction < 1.0:
            raise ValueError("contraction must be between zero and one")
        if not np.isfinite(params.profit_target) or params.profit_target <= 0:
            raise ValueError("profit_target must be finite and positive")
        if not 5 <= params.hold_bars <= 10:
            raise ValueError("hold_bars must be between 5 and 10")

        out = pd.DataFrame(index=data.index)
        returns = (data["close"] / data["close"].shift(1) - 1.0).replace([np.inf, -np.inf], np.nan)
        down = returns.lt(0).fillna(False)
        streak = pd.Series(0, index=data.index, dtype=int)
        uninterrupted = pd.Series(True, index=data.index)
        for lag in range(7):
            uninterrupted = uninterrupted & down.shift(lag, fill_value=False)
            streak = streak + uninterrupted.astype(int)

        out["return"] = returns
        out["previous_return"] = returns.shift(1)
        out["earlier_return"] = returns.shift(2)
        out["down_streak"] = streak
        out["autocorrelation"] = returns.rolling(6, min_periods=6).corr(returns.shift(1)).replace([np.inf, -np.inf], np.nan).fillna(0.0)
        out["decelerating"] = (
            returns.abs().le(params.contraction * returns.shift(1).abs())
            & returns.shift(1).abs().le(params.contraction * returns.shift(2).abs())
        ).fillna(False)
        out["entry"] = (
            streak.ge(params.min_streak)
            & out["autocorrelation"].gt(0.0)
            & out["decelerating"]
        ).fillna(False)
        return out

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: ShockwaveParams) -> SignalFrame:
        opens = data["open"].to_numpy(dtype=float)
        closes = data["close"].to_numpy(dtype=float)
        if not (np.isfinite(opens).all() and np.isfinite(closes).all() and (opens > 0).all() and (closes > 0).all()):
            raise ValueError("open and close prices must be finite and positive")
        entries = indicators["entry"].fillna(False).to_numpy(dtype=bool)
        decisions = np.zeros(len(data), dtype=int)
        pending = False
        active = False
        entry_price = 0.0
        entry_bar = -1

        for i in range(len(data)):
            if pending:
                # ponytail: use the observed entry open; fill-aware targets need execution feedback.
                entry_price = opens[i]
                entry_bar = i
                active = True
                pending = False

            if active:
                reached_target = closes[i] >= entry_price * (1.0 + params.profit_target)
                expired = i - entry_bar + 1 >= params.hold_bars
                if reached_target or expired:
                    active = False
                else:
                    decisions[i] = 1
            elif i >= self.warmup_bars(params) - 1 and entries[i]:
                decisions[i] = 1
                pending = True

        df = pd.DataFrame(index=data.index)
        df["signal"] = decisions
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = 1.0
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def self_check() -> None:
    strategy = GeneratedStrategy()
    params = ShockwaveParams(hold_bars=5)
    index = pd.date_range("2020-01-01", periods=16, freq="B")
    data = pd.DataFrame({"open": 100.0, "high": 104.0, "low": 99.0, "close": 100.0, "volume": 1000.0}, index=index)
    derived = strategy.indicators(data, params)
    assert derived.index.equals(data.index)
    assert not derived["entry"].any()
    assert strategy.warmup_bars(params) <= 10
    entries = pd.DataFrame({"entry": False}, index=index)
    entries.iloc[7, 0] = True
    timed = strategy.generate_signals(data, entries, None, params).data
    assert timed["signal"].tolist() == [0] * 8 + [1] * 5 + [0] * 3
    data.loc[index[9], "close"] = 103.0
    targeted = strategy.generate_signals(data, entries, None, params).data
    assert targeted["signal"].tolist() == [0] * 8 + [1] * 2 + [0] * 6
    assert targeted["size"].gt(0).all()


if __name__ == "__main__":
    self_check()
