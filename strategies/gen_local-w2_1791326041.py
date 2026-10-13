from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class ExpirationParams:
    min_streak: int = 3
    profit_target: float = 0.04
    hold_bars: int = 18


class GeneratedStrategy(BaseStrategy[ExpirationParams]):
    strategy_id = "gen_local-w2_1791326041"

    @classmethod
    def params_type(cls):
        return ExpirationParams

    @staticmethod
    def warmup_bars(params: ExpirationParams) -> int:
        return 11

    @staticmethod
    def indicators(data: pd.DataFrame, params: ExpirationParams) -> pd.DataFrame:
        if not 1 <= params.min_streak <= 6:
            raise ValueError("min_streak must be between 1 and 6")
        if params.profit_target <= 0 or params.hold_bars < 1:
            raise ValueError("profit_target and hold_bars must be positive")
        out = pd.DataFrame(index=data.index)
        day = data.index.day
        first_weekday = (data.index.dayofweek - (day - 1)) % 7
        third_friday = 15 + (4 - first_weekday) % 7
        out["expiration_offset"] = day - third_friday
        out["return"] = data["close"].pct_change(fill_method=None).replace([np.inf, -np.inf], np.nan)
        selling = out["return"].lt(0) & out["expiration_offset"].between(-9, -1)
        rising = out["return"].gt(0)
        out["selling_streak"] = selling.astype(int).groupby((~selling).cumsum()).cumsum()
        out["rising_streak"] = rising.astype(int).groupby((~rising).cumsum()).cumsum()
        out["prior_pressure"] = out["selling_streak"].rolling(8, min_periods=8).max().shift(2).fillna(0.0)
        out["entry"] = (
            out["expiration_offset"].between(0, 7)
            & out["expiration_offset"].shift(1).ge(0)
            & out["prior_pressure"].ge(params.min_streak)
            & out["rising_streak"].eq(2)
        ).fillna(False)
        return out

    @staticmethod
    def generate_signals(data: pd.DataFrame, indicators: pd.DataFrame,
                         ctx: StrategyContext, params: ExpirationParams) -> SignalFrame:
        if params.profit_target <= 0 or params.hold_bars < 1:
            raise ValueError("profit_target and hold_bars must be positive")
        decisions = np.zeros(len(data), dtype=int)
        candidates = indicators["entry"].fillna(False).to_numpy(dtype=bool)
        opens = data["open"].to_numpy(dtype=float)
        closes = data["close"].to_numpy(dtype=float)
        pending = False
        holding = False
        entry_price = 0.0
        entry_bar = -1
        warmup = GeneratedStrategy.warmup_bars(params)
        for i in range(len(data)):
            if pending:
                pending = False
                if np.isfinite(opens[i]) and opens[i] > 0:
                    holding = True
                    entry_price = opens[i]
                    entry_bar = i
            if holding:
                target_hit = np.isfinite(closes[i]) and closes[i] >= entry_price * (1.0 + params.profit_target)
                time_up = i - entry_bar + 1 >= params.hold_bars
                if target_hit or time_up:
                    holding = False
                else:
                    decisions[i] = 1
            elif i >= warmup - 1 and candidates[i] and np.isfinite(closes[i]) and closes[i] > 0:
                pending = True
                decisions[i] = 1
        df = pd.DataFrame({"signal": decisions, "size": 0.95}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check():
    index = pd.bdate_range("2020-01-01", periods=20)
    data = pd.DataFrame({"open": 100.0, "high": 101.0, "low": 99.0,
                         "close": 100.0, "volume": 1000.0}, index=index)
    indicators = pd.DataFrame({"entry": False}, index=index)
    indicators.iloc[11, 0] = True
    params = ExpirationParams(hold_bars=3)
    timed = GeneratedStrategy.generate_signals(data, indicators, None, params).data
    assert timed["signal"].iloc[11:17].tolist() == [0, 1, 1, 1, 0, 0]
    data.iloc[13, data.columns.get_loc("close")] = 105.0
    targeted = GeneratedStrategy.generate_signals(data, indicators, None, params).data
    assert targeted["signal"].iloc[11:17].tolist() == [0, 1, 1, 0, 0, 0]
    assert set(targeted["signal"]).issubset({0, 1})
    assert targeted["size"].gt(0).all()


if __name__ == "__main__":
    _self_check()
