from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class OrdinalParams:
    lookback: int = 126
    hold_bars: int = 4


class GeneratedStrategy(BaseStrategy[OrdinalParams]):
    strategy_id = "gen_local_1790982900"

    @classmethod
    def params_type(cls):
        return OrdinalParams

    def warmup_bars(self, params: OrdinalParams) -> int:
        if params.lookback < 12 or not 3 <= params.hold_bars <= 5:
            raise ValueError("lookback must be >= 12; hold_bars must be 3-5")
        return int(params.lookback + params.hold_bars + 3)

    def indicators(self, data: pd.DataFrame, params: OrdinalParams) -> pd.DataFrame:
        warmup = self.warmup_bars(params)
        result = pd.DataFrame(index=data.index)
        log_close = np.log(data["close"].where(data["close"] > 0))
        returns = log_close.diff().replace([np.inf, -np.inf], np.nan)
        a, b, c = returns.shift(2), returns.shift(1), returns
        distinct = a.notna() & b.notna() & c.notna() & a.ne(b) & a.ne(c) & b.ne(c)
        pattern = (4 * a.gt(b).astype(int) + 2 * a.gt(c).astype(int) + b.gt(c).astype(int)).where(distinct)
        result["log_return"] = returns
        result["pattern"] = pattern
        # Each training label is attached only after its entire holding horizon ends.
        result["source_pattern"] = pattern.shift(params.hold_bars)
        result["successor_pattern"] = pattern.shift(params.hold_bars - 1)
        result["realized_return"] = log_close.diff(params.hold_bars).replace([np.inf, -np.inf], np.nan)
        source = result["source_pattern"]
        successor = result["successor_pattern"]
        outcome = result["realized_return"]
        valid = source.notna() & successor.notna() & outcome.notna()
        codes = (0, 1, 3, 4, 6, 7)
        for code in codes:
            sample = valid & source.eq(code)
            count = sample.astype(float).rolling(params.lookback, min_periods=params.lookback).sum()
            mean = outcome.where(sample).rolling(params.lookback, min_periods=6).mean()
            std = outcome.where(sample).rolling(params.lookback, min_periods=6).std()
            concentration = pd.Series(0.0, index=data.index)
            for next_code in codes:
                hits = (sample & successor.eq(next_code)).astype(float).rolling(params.lookback, min_periods=params.lookback).sum()
                probability = hits.div(count.replace(0.0, np.nan))
                result[f"transition_{code}_{next_code}"] = probability
                concentration = concentration + probability.pow(2)
            result[f"count_{code}"] = count
            result[f"mean_{code}"] = mean
            result[f"std_{code}"] = std
            result[f"complexity_{code}"] = (1.0 - concentration).clip(0.0, 1.0)
        result["forecast"] = np.nan
        result["complexity"] = np.nan
        for code in codes:
            current = pattern.eq(code)
            result.loc[current, "forecast"] = result[f"mean_{code}"].div(result[f"std_{code}"].replace(0.0, np.nan)).loc[current]
            result.loc[current, "complexity"] = result.loc[current, f"complexity_{code}"]
        result["ready"] = np.arange(len(data)) >= warmup - 1
        return result.replace([np.inf, -np.inf], np.nan)

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: OrdinalParams) -> SignalFrame:
        self.warmup_bars(params)
        forecast = indicators["forecast"]
        eligible = (indicators["ready"].fillna(False) & forecast.notna() & indicators["complexity"].notna()
                    & indicators["complexity"].le(0.55) & forecast.abs().ge(0.20))
        entries = np.where(eligible, np.sign(forecast.fillna(0.0)), 0).astype(int)
        signals = np.zeros(len(data), dtype=int)
        position, exit_at = 0, -1
        for i, entry in enumerate(entries):
            if position != 0:
                if i == exit_at:
                    position = 0
            elif entry != 0:
                position = int(entry)
                exit_at = i + params.hold_bars
            signals[i] = position
        df = pd.DataFrame({"signal": signals, "size": 0.95}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


if __name__ == "__main__":
    strategy = GeneratedStrategy()
    params = OrdinalParams()
    index = pd.date_range("2020-01-01", periods=16, freq="D")
    prices = pd.DataFrame({"open": 100.0, "high": 101.0, "low": 99.0, "close": 100.0, "volume": 1000.0}, index=index)
    features = pd.DataFrame({"forecast": [1.0] * 3 + [-1.0] * 13, "complexity": 0.0, "ready": True}, index=index)
    ctx = StrategyContext(symbol="SPY", timeframe="1d", warmup_bars=strategy.warmup_bars(params))
    for holding in (3, 4, 5):
        params.hold_bars = holding
        actual = strategy.generate_signals(prices, features, ctx, params).data["signal"].tolist()
        assert actual[:holding + 2] == [0] + [1] * holding + [0]
        assert actual[holding + 2:2 * holding + 3] == [-1] * holding + [0]
    params.hold_bars = 4
    flat = strategy.indicators(prices, params)
    assert strategy.generate_signals(prices, flat, ctx, params).data["signal"].eq(0).all()
