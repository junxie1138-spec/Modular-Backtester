from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class VisibilityParams:
    window: int = 5
    consensus: float = 0.5


class GeneratedStrategy(BaseStrategy[VisibilityParams]):
    strategy_id = "gen_local-w2_1791289359"

    @classmethod
    def params_type(cls):
        return VisibilityParams

    def warmup_bars(self, params):
        return params.window

    def indicators(self, data, params):
        if not isinstance(params.window, int) or params.window < 3:
            raise ValueError("window must be an integer >= 3")
        if not np.isfinite(params.consensus) or not 0 < params.consensus <= 1:
            raise ValueError("consensus must be in (0, 1]")
        price = data["close"].astype(float)
        log_close = np.log(price.where((price > 0) & np.isfinite(price)))
        valid = log_close.rolling(params.window).count().eq(params.window)
        up = pd.Series(0.0, index=data.index)
        down = pd.Series(0.0, index=data.index)
        # A triangle connects t-lag, t-1 and t; each edge must be visible.
        # ponytail: O(window**2) shifted comparisons; optimize if windows grow large.
        for lag in range(2, params.window):
            anchor = log_close.shift(lag)
            upper = valid.copy()
            lower = valid.copy()
            for middle in range(1, lag):
                between = log_close.shift(middle)
                chord = log_close + (anchor - log_close) * (middle / lag)
                upper &= between < chord
                lower &= between > chord
                if middle > 1:
                    prior_chord = log_close.shift(1) + (
                        anchor - log_close.shift(1)
                    ) * ((middle - 1) / (lag - 1))
                    upper &= between < prior_chord
                    lower &= between > prior_chord
            rising = (log_close > log_close.shift(1)) & (
                log_close.shift(1) > anchor
            )
            falling = (log_close < log_close.shift(1)) & (
                log_close.shift(1) < anchor
            )
            up += (upper & rising).astype(float)
            down += (lower & falling).astype(float)
        score = (up - down) / (params.window - 2)
        return pd.DataFrame({
            "log_close": log_close,
            "valid": valid,
            "up_triangles": up,
            "down_triangles": down,
            "score": score.where(valid),
        }, index=data.index)

    def generate_signals(self, data, indicators, ctx, params):
        score = indicators["score"]
        condition = np.select(
            [score.ge(params.consensus), score.le(-params.consensus)],
            [1, -1], default=0,
        )
        signals = np.zeros(len(data), dtype=int)
        position = 0
        for bar, candidate in enumerate(condition):
            if position == 0:
                position = int(candidate)
            elif candidate == -position:
                position = 0
            signals[bar] = position
        df = pd.DataFrame({"signal": signals, "size": 0.95}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check():
    strategy = GeneratedStrategy()
    params = VisibilityParams()
    index = pd.date_range("2020-01-01", periods=12)
    close = 100 * np.exp(0.001 * np.arange(12, dtype=float) ** 2)
    data = pd.DataFrame({
        "open": close, "high": close, "low": close,
        "close": close, "volume": 1000.0,
    }, index=index)
    ctx = StrategyContext("SPY", "1d", strategy.warmup_bars(params))
    indicators = strategy.indicators(data, params)
    assert indicators["score"].iloc[params.window - 1:].eq(1.0).all()
    mirrored = data.copy()
    mirrored["close"] = 10000.0 / data["close"]
    assert strategy.indicators(mirrored, params)["score"].iloc[
        params.window - 1:
    ].eq(-1.0).all()
    evidence = pd.DataFrame({"score": [1, 0, -1, -1, 0, 1, 1]}, index=index[:7])
    signals = strategy.generate_signals(data.iloc[:7], evidence, ctx, params).data
    assert signals["signal"].tolist() == [0, 1, 1, 0, -1, -1, 0]
    assert signals["size"].gt(0).all()
    for end in range(params.window, len(data) + 1):
        prefix = strategy.indicators(data.iloc[:end], params)
        pd.testing.assert_frame_equal(prefix, indicators.iloc[:end])


if __name__ == "__main__":
    _self_check()
