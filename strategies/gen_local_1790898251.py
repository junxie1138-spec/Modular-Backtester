from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class RangeTideParams:
    range_lookback: int = 5
    season_lookback: int = 126
    range_percentile: float = 0.25
    profit_target: float = 0.025
    max_hold: int = 8


class GeneratedStrategy(BaseStrategy[RangeTideParams]):
    strategy_id = "gen_local_1790898251"

    @classmethod
    def params_type(cls) -> type[RangeTideParams]:
        return RangeTideParams

    def warmup_bars(self, params: RangeTideParams) -> int:
        return int(params.range_lookback + params.season_lookback + 2)

    def indicators(self, data: pd.DataFrame, params: RangeTideParams) -> pd.DataFrame:
        if params.range_lookback < 1 or params.season_lookback < 30:
            raise ValueError("Lookbacks must be positive; season_lookback must be at least 30.")
        if not 0.0 < params.range_percentile < 1.0:
            raise ValueError("range_percentile must lie strictly between zero and one.")
        if params.profit_target <= 0.0 or params.max_hold < 1:
            raise ValueError("profit_target and max_hold must be positive.")
        out = pd.DataFrame(index=data.index)
        out["range"] = (data["high"] - data["low"]).where(data["high"] >= data["low"])
        out["prior_range_mean"] = out["range"].shift(1).rolling(params.range_lookback, min_periods=params.range_lookback).mean()
        out["range_ratio"] = out["range"].div(out["prior_range_mean"].where(out["prior_range_mean"] > 0.0))
        out["range_ratio"] = out["range_ratio"].replace([np.inf, -np.inf], np.nan)
        out["is_wednesday"] = data.index.dayofweek == 2
        historical_wednesdays = out["range_ratio"].where(out["is_wednesday"]).shift(1)
        out["weekday_threshold"] = historical_wednesdays.rolling(params.season_lookback, min_periods=4).quantile(params.range_percentile)
        out["bullish_close"] = data["close"].ge(data["open"])
        out["entry"] = (out["is_wednesday"] & out["bullish_close"] & out["range_ratio"].le(out["weekday_threshold"]) & data["close"].gt(0.0)).fillna(False)
        return out

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: RangeTideParams) -> SignalFrame:
        entries = indicators["entry"].fillna(False).to_numpy(dtype=bool)
        opens = data["open"].to_numpy(dtype=float)
        closes = data["close"].to_numpy(dtype=float)
        signals = np.zeros(len(data), dtype=int)
        active = False
        entry_price = np.nan
        held_bars = 0
        warmup = self.warmup_bars(params)

        for i in range(len(data)):
            if active:
                if np.isnan(entry_price):
                    entry_price = opens[i]
                held_bars += 1
                valid_entry = np.isfinite(entry_price) and entry_price > 0.0
                target_hit = valid_entry and np.isfinite(closes[i]) and closes[i] >= entry_price * (1.0 + params.profit_target)
                if not valid_entry or target_hit or held_bars >= params.max_hold:
                    active = False
                    entry_price = np.nan
                    held_bars = 0
                else:
                    signals[i] = 1
            elif i >= warmup and entries[i]:
                active = True
                entry_price = np.nan
                held_bars = 0
                signals[i] = 1

        df = pd.DataFrame({"signal": signals, "size": 1.0}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check() -> None:
    index = pd.bdate_range("2020-01-01", periods=160)
    data = pd.DataFrame({"open": 100.0, "high": 101.0, "low": 99.0, "close": 100.0, "volume": 1000.0}, index=index)
    params = RangeTideParams(max_hold=8)
    strategy = GeneratedStrategy()
    derived = strategy.indicators(data, params)
    assert derived.index.equals(data.index)
    assert not derived["entry"].iloc[:params.range_lookback].any()
    derived["entry"] = False
    start = strategy.warmup_bars(params)
    derived.iloc[start, derived.columns.get_loc("entry")] = True
    timed = strategy.generate_signals(data, derived, None, params).data
    assert timed["signal"].iloc[start] == 0
    assert timed["signal"].iloc[start + 1:start + 9].eq(1).all()
    assert timed["signal"].iloc[start + 9] == 0
    data.iloc[start + 2, data.columns.get_loc("close")] = 103.0
    targeted = strategy.generate_signals(data, derived, None, params).data
    assert targeted["signal"].iloc[start + 2] == 1
    assert targeted["signal"].iloc[start + 3] == 0
    assert targeted["signal"].isin([0, 1]).all()
    assert targeted["size"].gt(0.0).all()


if __name__ == "__main__":
    _self_check()
