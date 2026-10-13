from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class Params:
    lookback: int = 24
    atr_multiple: float = 2.0

    def __post_init__(self):
        if self.lookback < 8 or not np.isfinite(self.atr_multiple) or self.atr_multiple <= 0:
            raise ValueError("lookback must be >= 8 and atr_multiple must be finite and positive")


class GeneratedStrategy(BaseStrategy[Params]):
    strategy_id = "gen_local_1791336888"

    @classmethod
    def params_type(cls):
        return Params

    def warmup_bars(self, params: Params) -> int:
        return max(params.lookback + 4, 18)

    def indicators(self, data: pd.DataFrame, params: Params) -> pd.DataFrame:
        out = pd.DataFrame(index=data.index)
        span = (data["high"] - data["low"]).replace(0, np.nan)
        location = (data["close"] - data["low"]) / span
        a, b, c = location.shift(2), location.shift(1), location
        valid = a.notna() & b.notna() & c.notna() & a.ne(b) & a.ne(c) & b.ne(c)
        pattern = (4 * a.gt(b).astype(int) + 2 * a.gt(c).astype(int) + b.gt(c).astype(int)).where(valid)
        edge = pattern.shift(1) * 8 + pattern
        complexity = pd.Series(0.0, index=data.index)
        novel = pd.Series(False, index=data.index)
        for left in (0, 1, 3, 4, 6, 7):
            for right in (0, 1, 3, 4, 6, 7):
                match = edge.eq(left * 8 + right)
                count = match.astype(float).rolling(params.lookback, min_periods=params.lookback).sum()
                complexity += count.gt(0).astype(float)
                novel |= match & count.shift(1).eq(0)
        ready = edge.notna().rolling(params.lookback, min_periods=params.lookback).sum().eq(params.lookback)
        complexity = complexity.where(ready)
        previous_close = data["close"].shift(1)
        true_range = pd.concat(
            [span, (data["high"] - previous_close).abs(), (data["low"] - previous_close).abs()],
            axis=1,
        ).max(axis=1).where(previous_close.notna())
        atr = true_range.rolling(14, min_periods=14).mean().replace(0, np.nan)
        spike = true_range.gt(3.0 * atr)
        refractory = spike.astype(float).rolling(4, min_periods=4).max().eq(1)
        upper = data["high"].rolling(params.lookback, min_periods=params.lookback).max().shift(1)
        lower = data["low"].rolling(params.lookback, min_periods=params.lookback).min().shift(1)
        gate = novel & complexity.gt(complexity.shift(1)) & complexity.ge(params.lookback / 2.0)
        gate &= atr.notna() & ~refractory
        entry = pd.Series(0, index=data.index, dtype=int)
        entry.loc[gate & data["close"].gt(upper)] = 1
        entry.loc[gate & data["close"].lt(lower)] = -1
        out["close_location"] = location
        out["ordinal_pattern"] = pattern
        out["ordinal_transition"] = edge
        out["transition_complexity"] = complexity
        out["novel_transition"] = novel
        out["true_range"] = true_range
        out["atr"] = atr
        out["refractory"] = refractory
        out["upper"] = upper
        out["lower"] = lower
        out["entry"] = entry
        return out

    def generate_signals(
        self, data: pd.DataFrame, indicators: pd.DataFrame,
        ctx: StrategyContext, params: Params,
    ) -> SignalFrame:
        signals = np.zeros(len(data), dtype=int)
        side, age = 0, 0
        peak, trough = -np.inf, np.inf
        high_stop, short_stop = -np.inf, np.inf
        closes = data["close"].to_numpy(dtype=float)
        atrs = indicators["atr"].to_numpy(dtype=float)
        entries = indicators["entry"].to_numpy(dtype=int)
        for i in range(len(data)):
            close, atr = closes[i], atrs[i]
            if not np.isfinite(close) or not np.isfinite(atr) or atr <= 0:
                side = 0
                continue
            if side:
                age += 1
                peak = max(peak, close)
                trough = min(trough, close)
                high_stop = max(high_stop, peak - params.atr_multiple * atr)
                short_stop = min(short_stop, trough + params.atr_multiple * atr)
                # The high-water drawdown exit applies to both directions.
                # Shorts additionally have a ratcheting adverse-move stop.
                if close < high_stop or (side == -1 and close > short_stop) or age >= 10:
                    side = 0
                    continue
            elif entries[i]:
                side, age = int(entries[i]), 0
                # Entry fills next bar; extrema begin with that bar's close.
                peak, trough = -np.inf, np.inf
                high_stop, short_stop = -np.inf, np.inf
            signals[i] = side
        df = pd.DataFrame({"signal": signals, "size": 0.95}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check():
    params = Params(lookback=8, atr_multiple=2.0)
    index = pd.date_range("2020-01-01", periods=7, freq="D")
    data = pd.DataFrame({"close": [10, 11, 13, 12.5, 14, 11, 11]}, index=index)
    ind = pd.DataFrame({"atr": 1.0, "entry": [1, 0, 0, 0, 0, 0, 0]}, index=index)
    result = GeneratedStrategy.generate_signals(None, data, ind, None, params).data
    assert result["signal"].tolist() == [0, 1, 1, 1, 1, 1, 0]
    # ATR expansion cannot lower an already-ratcheted stop.
    ind["atr"] = [1, 1, 1, 10, 10, 10, 10]
    data["close"] = [10, 11, 13, 10, 10, 10, 10]
    result = GeneratedStrategy.generate_signals(None, data, ind, None, params).data
    assert result["signal"].tolist() == [0, 1, 1, 1, 0, 0, 0]
    ind["atr"] = 1.0
    ind["entry"] = [-1, 0, 0, 0, 0, 0, 0]
    data["close"] = [10, 10, 9, 11.5, 11.5, 11.5, 11.5]
    result = GeneratedStrategy.generate_signals(None, data, ind, None, params).data
    assert result["signal"].tolist() == [0, -1, -1, -1, 0, 0, 0]


if __name__ == "__main__":
    _self_check()
