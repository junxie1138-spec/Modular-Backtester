from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class Params:
    lookback: int = 12
    profit_target: float = 0.012


class GeneratedStrategy(BaseStrategy[Params]):
    strategy_id = "gen_local_1791235716"

    @classmethod
    def params_type(cls) -> type[Params]:
        return Params

    @staticmethod
    def warmup_bars(params: Params) -> int:
        if params.lookback < 2:
            raise ValueError("lookback must be at least 2")
        if not np.isfinite(params.profit_target) or params.profit_target <= 0:
            raise ValueError("profit_target must be finite and positive")
        return int(params.lookback) + 2

    @staticmethod
    def indicators(data: pd.DataFrame, params: Params) -> pd.DataFrame:
        GeneratedStrategy.warmup_bars(params)
        out = pd.DataFrame(index=data.index)
        close = data["close"].where(data["close"] > 0)
        out["return"] = close.pct_change(fill_method=None).replace([np.inf, -np.inf], np.nan)
        out["prior_return"] = out["return"].shift(1)
        window = out["prior_return"].rolling(params.lookback, min_periods=params.lookback)
        out["prior_max"] = window.max()
        out["prior_min"] = window.min()
        out["prior_net"] = window.sum()
        out["prior_gross"] = out["prior_return"].abs().rolling(params.lookback, min_periods=params.lookback).sum()
        out["cancellation_ratio"] = out["prior_net"].abs().div(out["prior_gross"].where(out["prior_gross"] > 0))
        balanced = out["cancellation_ratio"].le(0.20) & out["prior_max"].gt(0) & out["prior_min"].lt(0)
        long_entry = balanced & out["return"].gt(out["prior_max"])
        short_entry = balanced & out["return"].lt(out["prior_min"])
        out["breakout"] = np.select([long_entry, short_entry], [1, -1], default=0).astype(int)
        return out

    @staticmethod
    def generate_signals(data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: Params) -> SignalFrame:
        GeneratedStrategy.warmup_bars(params)
        opens = data["open"].to_numpy(dtype=float)
        closes = data["close"].to_numpy(dtype=float)
        setups = indicators["breakout"].fillna(0).to_numpy(dtype=int)
        decisions = np.zeros(len(data), dtype=int)
        position = 0
        entry_price = 0.0
        held_bars = 0
        for i in range(len(data)):
            if not np.isfinite(closes[i]) or closes[i] <= 0:
                position = 0
                entry_price = 0.0
                held_bars = 0
                continue
            if position != 0:
                # The previous close's entry decision fills at this bar's open.
                if entry_price == 0.0:
                    if not np.isfinite(opens[i]) or opens[i] <= 0:
                        position = 0
                        held_bars = 0
                        continue
                    entry_price = opens[i]
                held_bars += 1
                gain = position * (closes[i] / entry_price - 1.0)
                if gain >= params.profit_target or held_bars >= 4:
                    position = 0
                    entry_price = 0.0
                    held_bars = 0
            elif setups[i] != 0:
                position = int(setups[i])
                entry_price = 0.0
                held_bars = 0
            decisions[i] = position
        df = pd.DataFrame({"signal": decisions, "size": 1.0}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check() -> None:
    index = pd.date_range("2020-01-01", periods=7)
    data = pd.DataFrame({"open": 100.0, "high": 103.0, "low": 97.0, "close": 100.0, "volume": 1000.0}, index=index)
    indicators = pd.DataFrame({"breakout": [1, 0, 0, 0, 0, 0, 0]}, index=index)
    result = GeneratedStrategy.generate_signals(data, indicators, None, Params()).data
    assert result["signal"].tolist() == [0, 1, 1, 1, 1, 0, 0]
    for direction in (1, -1):
        indicators["breakout"] = [direction, 0, 0, 0, 0, 0, 0]
        data["close"] = 100.0
        data.loc[index[1], "close"] = 100.0 + direction * 2.0
        result = GeneratedStrategy.generate_signals(data, indicators, None, Params()).data
        assert result["signal"].tolist() == [0, direction, 0, 0, 0, 0, 0]
        assert result["size"].gt(0).all()
    derived = GeneratedStrategy.indicators(data, Params())
    assert derived["breakout"].eq(0).all()
    assert GeneratedStrategy.warmup_bars(Params()) >= Params().lookback + 2


if __name__ == "__main__":
    _self_check()
