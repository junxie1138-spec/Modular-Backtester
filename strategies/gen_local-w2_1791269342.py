from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class OrdinalReversionParams:
    pattern_window: int = 6
    channel_window: int = 5
    smoothing_window: int = 3
    complexity_ceiling: float = 0.65
    downside_level: float = 0.35
    size: float = 0.95

    def __post_init__(self):
        if min(self.pattern_window, self.channel_window, self.smoothing_window) < 2:
            raise ValueError("Windows must be at least two bars.")
        if not 0.0 <= self.complexity_ceiling <= 1.0:
            raise ValueError("complexity_ceiling must be in [0, 1].")
        if not 0.0 < self.downside_level <= 1.0 or not 0.0 < self.size <= 1.0:
            raise ValueError("downside_level and size must be in (0, 1].")


class GeneratedStrategy(BaseStrategy[OrdinalReversionParams]):
    strategy_id = "gen_local-w2_1791269342"

    @classmethod
    def params_type(cls):
        return OrdinalReversionParams

    def warmup_bars(self, params):
        return int(max(params.pattern_window + 2,
                       params.channel_window + params.smoothing_window - 1))

    def indicators(self, data, params):
        out = pd.DataFrame(index=data.index)
        close = data["close"].replace([np.inf, -np.inf], np.nan)
        out["close_lag2"] = close.shift(2)
        out["close_lag1"] = close.shift(1)
        a, b = out["close_lag2"], out["close_lag1"]
        valid = close.notna() & a.notna() & b.notna()
        # Equal prices retain chronological order, so ties are deterministic.
        out["pattern"] = (
            4 * a.le(b).astype(int)
            + 2 * a.le(close).astype(int)
            + b.le(close).astype(int)
        ).astype(float).where(valid)
        collision = pd.Series(0.0, index=data.index)
        for pattern in (0, 1, 3, 4, 6, 7):
            frequency = out["pattern"].eq(pattern).astype(float).where(valid)
            column = "frequency_" + str(pattern)
            out[column] = frequency.rolling(
                params.pattern_window, min_periods=params.pattern_window
            ).mean()
            collision = collision + out[column].pow(2)
        out["complexity"] = ((1.0 - collision) / (5.0 / 6.0)).clip(0.0, 1.0)
        out["channel_high"] = data["high"].replace(
            [np.inf, -np.inf], np.nan
        ).rolling(params.channel_window, min_periods=params.channel_window).max()
        out["channel_low"] = data["low"].replace(
            [np.inf, -np.inf], np.nan
        ).rolling(params.channel_window, min_periods=params.channel_window).min()
        out["channel_width"] = out["channel_high"] - out["channel_low"]
        width = out["channel_width"].where(out["channel_width"] > 0.0)
        out["location"] = 2.0 * (close - out["channel_low"]) / width - 1.0
        out["displacement"] = out["location"].rolling(
            params.smoothing_window, min_periods=params.smoothing_window
        ).mean()
        return out.replace([np.inf, -np.inf], np.nan)

    def generate_signals(self, data, indicators, ctx, params):
        entry = (
            indicators["complexity"].le(params.complexity_ceiling)
            & indicators["displacement"].le(-params.downside_level)
            & indicators["complexity"].notna()
            & indicators["displacement"].notna()
        )
        df = pd.DataFrame(index=data.index)
        # Exit only when the combined entry condition flips to false.
        df["signal"] = entry.astype(int)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = float(params.size)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check():
    params = OrdinalReversionParams()
    strategy = GeneratedStrategy()
    index = pd.date_range("2020-01-01", periods=30, freq="D")
    close = pd.Series(np.arange(130.0, 100.0, -1.0), index=index)
    data = pd.DataFrame({
        "open": close, "high": close + 0.5, "low": close - 0.5,
        "close": close, "volume": 1000.0,
    }, index=index)
    ctx = StrategyContext(symbol="SPY", timeframe="1d",
                          warmup_bars=strategy.warmup_bars(params))
    derived = strategy.indicators(data, params)
    assert derived["complexity"].first_valid_index() == index[params.pattern_window + 1]
    assert derived["complexity"].dropna().eq(0.0).all()
    signals = strategy.generate_signals(data, derived, ctx, params).data
    assert signals["signal"].iloc[:ctx.warmup_bars].eq(0).all()
    assert signals["signal"].iloc[-1] == 1
    probe = pd.DataFrame({
        "complexity": [np.nan, 0.1, 0.1, 0.9, 0.1, 0.1],
        "displacement": [-0.8, -0.8, -0.8, -0.8, 0.8, -0.8],
    }, index=index[:6])
    result = strategy.generate_signals(data.iloc[:6], probe, ctx, params).data
    assert result["signal"].tolist() == [0, 0, 1, 1, 0, 0]
    assert result["size"].gt(0.0).all()
    flat = data.copy()
    flat[["open", "high", "low", "close"]] = 100.0
    flat_signals = strategy.generate_signals(
        flat, strategy.indicators(flat, params), ctx, params
    ).data
    assert flat_signals["signal"].eq(0).all()


if __name__ == "__main__":
    _self_check()
