from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class OrdinalComplexityParams:
    ordinal_window: int = 24
    mean_window: int = 20
    complexity_min: float = 0.035
    stretch: float = 1.2
    profit_target: float = 0.015
    max_hold_bars: int = 8


class GeneratedStrategy(BaseStrategy[OrdinalComplexityParams]):
    strategy_id = "gen_local-w2_1791282447"

    @classmethod
    def params_type(cls):
        return OrdinalComplexityParams

    def warmup_bars(self, params: OrdinalComplexityParams) -> int:
        return int(max(params.ordinal_window + 3, params.mean_window) + 1)

    def indicators(self, data: pd.DataFrame, params: OrdinalComplexityParams) -> pd.DataFrame:
        if params.ordinal_window < 3 or params.mean_window < 2:
            raise ValueError("Lookback windows must be at least 3 and 2.")
        if not (0 <= params.complexity_min <= 1 and params.stretch > 0
                and params.profit_target > 0 and params.max_hold_bars >= 1):
            raise ValueError("Invalid entry or exit parameters.")
        out = pd.DataFrame(index=data.index)
        out["return"] = data["close"].pct_change(fill_method=None)
        a, b, c = out["return"].shift(2), out["return"].shift(1), out["return"]
        valid = (np.isfinite(a) & np.isfinite(b) & np.isfinite(c)
                 & a.ne(b) & a.ne(c) & b.ne(c))
        out["pattern"] = ((a > b).astype(int) + 2 * (a > c).astype(int)
                          + 4 * (b > c).astype(int)).where(valid)
        probabilities = []
        for pattern in (0, 1, 3, 4, 6, 7):
            name = f"p_{pattern}"
            out[name] = (out["pattern"].eq(pattern).astype(float).where(valid)
                         .rolling(params.ordinal_window, min_periods=params.ordinal_window).mean())
            probabilities.append(name)
        p = out[probabilities]
        ready = p.notna().all(axis=1)
        out["entropy"] = (-(p * np.log(p.where(p > 0))).sum(axis=1)
                          / np.log(6.0)).where(ready)
        out["disequilibrium"] = ((p - 1.0 / 6.0).pow(2).sum(axis=1)
                                 * 6.0 / 5.0).where(ready)
        out["complexity"] = out["entropy"] * out["disequilibrium"]
        out["mean"] = data["close"].rolling(params.mean_window).mean()
        out["std"] = data["close"].rolling(params.mean_window).std(ddof=0)
        out["z"] = (data["close"] - out["mean"]) / out["std"].replace(0, np.nan)
        return out

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame,
                         ctx: StrategyContext, params: OrdinalComplexityParams) -> SignalFrame:
        eligible = indicators["complexity"].ge(params.complexity_min)
        candidates = pd.Series(0, index=data.index, dtype=int)
        candidates.loc[eligible & indicators["z"].le(-params.stretch)] = 1
        candidates.loc[eligible & indicators["z"].ge(params.stretch)] = -1
        confirmed = candidates.where(candidates.eq(candidates.shift(1)), 0).to_numpy()
        opens = data["open"].to_numpy(dtype=float)
        closes = data["close"].to_numpy(dtype=float)
        decisions = np.zeros(len(data), dtype=int)
        position, entry_bar = 0, -1
        entry_price = np.nan
        for i in range(len(data)):
            if position:
                # An entry decided yesterday first fills at today's open.
                if i == entry_bar:
                    entry_price = opens[i]
                if not (np.isfinite(entry_price) and entry_price > 0
                        and np.isfinite(closes[i]) and closes[i] > 0):
                    position = 0
                else:
                    gain = position * (closes[i] / entry_price - 1.0)
                    if gain >= params.profit_target or i - entry_bar + 1 >= params.max_hold_bars:
                        position = 0
                # Exits decided at this close fill at the next open.
                decisions[i] = position
                continue
            if confirmed[i] and np.isfinite(closes[i]) and closes[i] > 0:
                position = int(confirmed[i])
                entry_bar, entry_price = i + 1, np.nan
            decisions[i] = position
        df = pd.DataFrame({"signal": decisions, "size": 1.0}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check():
    index = pd.date_range("2020-01-01", periods=6)
    data = pd.DataFrame({"open": 100.0, "high": 101.0, "low": 99.0,
                         "close": 100.0, "volume": 1000.0}, index=index)
    indicators = pd.DataFrame({"complexity": 0.1, "z": [-2, -2, 0, 0, 0, 0]}, index=index)
    strategy = GeneratedStrategy()
    params = OrdinalComplexityParams(max_hold_bars=2)
    result = strategy.generate_signals(data, indicators, None, params).data
    assert result["signal"].tolist() == [0, 0, 1, 1, 0, 0]
    for direction in (1, -1):
        indicators["z"] = [-2 * direction, -2 * direction, 0, 0, 0, 0]
        data.loc[index[2], "close"] = 100 * (1 + direction * 0.03)
        result = strategy.generate_signals(data, indicators, None, params).data
        assert result["signal"].tolist() == [0, 0, direction, 0, 0, 0]
        assert result["size"].gt(0).all()
    t = np.arange(80, dtype=float)
    close = 100 + 0.1 * t + np.sin(t) + 0.3 * np.cos(t * 0.7)
    sample = pd.DataFrame({"close": close})
    full = strategy.indicators(sample, params)
    prefix = strategy.indicators(sample.iloc[:50], params)
    pd.testing.assert_frame_equal(full.iloc[:50], prefix)
    assert full["complexity"].dropna().between(0, 1).all()


if __name__ == "__main__":
    _self_check()
