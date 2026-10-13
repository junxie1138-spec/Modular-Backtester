from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class Params:
    fractional_order: float = 0.45
    fractional_window: int = 24
    baseline_window: int = 40
    compression_window: int = 8
    compression_ratio: float = 0.45
    release_z: float = 0.9
    atr_window: int = 14
    trailing_atr: float = 2.5
    max_hold_bars: int = 10


class GeneratedStrategy(BaseStrategy[Params]):
    strategy_id = "gen_local_1790980132"

    @classmethod
    def params_type(cls):
        return Params

    def warmup_bars(self, params: Params) -> int:
        return max(params.fractional_window + params.baseline_window + 7,
                   params.baseline_window + 7, params.atr_window + 1)

    def indicators(self, data: pd.DataFrame, params: Params) -> pd.DataFrame:
        if not (0.0 < params.fractional_order < 1.0):
            raise ValueError("fractional_order must lie between zero and one")
        if not (2 <= params.compression_window < params.baseline_window
                and params.fractional_window >= 2 and params.atr_window >= 2
                and params.max_hold_bars >= 1):
            raise ValueError("Invalid lookback or holding period")
        if not (0.0 < params.compression_ratio < 1.0
                and params.release_z > 0.0 and params.trailing_atr > 0.0):
            raise ValueError("Invalid compression, release or stop threshold")
        close = data["close"].astype(float)
        fractional = close.copy()
        weight = 1.0
        # ponytail: fixed-width fractional memory; increase the window for longer memory.
        for lag in range(1, params.fractional_window):
            weight *= (lag - 1.0 - params.fractional_order) / lag
            fractional = fractional + weight * close.shift(lag)
        history = fractional.shift(1).rolling(params.baseline_window)
        center = history.mean()
        scale = history.std(ddof=0).replace(0.0, np.nan)
        z = (fractional - center) / scale
        short_width = (data["high"].rolling(params.compression_window).max()
                       - data["low"].rolling(params.compression_window).min())
        long_width = (data["high"].rolling(params.baseline_window).max()
                      - data["low"].rolling(params.baseline_window).min())
        ratio = short_width / long_width.replace(0.0, np.nan)
        squeeze = ratio.le(params.compression_ratio) & ratio.notna()
        recent_squeeze = squeeze.shift(1, fill_value=False).astype(float).rolling(
            5, min_periods=5).max().eq(1.0)
        previous = close.shift(1)
        true_range = pd.concat([
            data["high"] - data["low"],
            (data["high"] - previous).abs(),
            (data["low"] - previous).abs()
        ], axis=1).max(axis=1)
        atr = true_range.rolling(params.atr_window).mean()
        return pd.DataFrame({
            "fractional_close": fractional,
            "fractional_center": center,
            "fractional_scale": scale,
            "fractional_z": z,
            "short_width": short_width,
            "long_width": long_width,
            "compression_ratio": ratio,
            "squeeze": squeeze,
            "recent_squeeze": recent_squeeze,
            "true_range": true_range,
            "atr": atr,
            "close_change": close.diff()
        }, index=data.index)

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame,
                         ctx: StrategyContext, params: Params) -> SignalFrame:
        z = indicators["fractional_z"]
        change = indicators["close_change"]
        valid = (np.isfinite(z) & np.isfinite(indicators["atr"])
                 & indicators["atr"].gt(0.0)
                 & np.isfinite(data["close"]))
        long_setup = (valid & indicators["recent_squeeze"].fillna(False)
                      & z.gt(params.release_z) & change.gt(0.0))
        short_setup = (valid & indicators["recent_squeeze"].fillna(False)
                       & z.lt(-params.release_z) & change.lt(0.0))
        confirmed_long = long_setup & long_setup.shift(1, fill_value=False)
        confirmed_short = short_setup & short_setup.shift(1, fill_value=False)
        entries = np.where(confirmed_long, 1, np.where(confirmed_short, -1, 0))
        closes = data["close"].to_numpy(dtype=float)
        atrs = indicators["atr"].to_numpy(dtype=float)
        signals = np.zeros(len(data), dtype=int)
        side = 0
        entry_bar = -1
        highest = -np.inf
        lowest = np.inf
        stop = 0.0
        for i in range(len(data)):
            if side:
                close, atr = closes[i], atrs[i]
                if not (np.isfinite(close) and np.isfinite(atr) and atr > 0.0):
                    side = 0
                else:
                    highest = max(highest, close)
                    lowest = min(lowest, close)
                    if side == 1:
                        stop = max(stop, highest - params.trailing_atr * atr)
                        stopped = close <= stop
                    else:
                        stop = min(stop, lowest + params.trailing_atr * atr)
                        stopped = close >= stop
                    if stopped or i - entry_bar + 1 >= params.max_hold_bars:
                        side = 0
                signals[i] = side
                continue
            if entries[i]:
                side = int(entries[i])
                entry_bar = i + 1
                highest, lowest = -np.inf, np.inf
                stop = -np.inf if side == 1 else np.inf
            signals[i] = side
        df = pd.DataFrame({"signal": signals, "size": 1.0}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check():
    strategy = GeneratedStrategy()
    params = Params(trailing_atr=2.0)
    index = pd.date_range("2020-01-01", periods=7)
    ctx = StrategyContext("SPY", "1d", strategy.warmup_bars(params))
    for direction in (1, -1):
        close = np.array([100, 101, 102, 105, 104, 102, 101], dtype=float)
        if direction == -1:
            close = 200.0 - close
        data = pd.DataFrame({"close": close}, index=index)
        derived = pd.DataFrame({
            "fractional_z": [0, 2 * direction, 2 * direction, 0, 0, 0, 0],
            "close_change": [0, direction, direction, 0, 0, 0, 0],
            "recent_squeeze": True,
            "atr": [1, 1, 1, 1, 3, 3, 3]
        }, index=index)
        result = strategy.generate_signals(data, derived, ctx, params).data
        assert result["signal"].tolist() == [
            0, 0, 0, direction, direction, direction, 0
        ]
        assert result["size"].gt(0.0).all()
    data = pd.DataFrame({
        "open": np.arange(120, dtype=float) + 100,
        "high": np.arange(120, dtype=float) + 101,
        "low": np.arange(120, dtype=float) + 99,
        "close": np.arange(120, dtype=float) + 100,
        "volume": 1000.0
    }, index=pd.date_range("2020-01-01", periods=120))
    derived = strategy.indicators(data, params)
    assert derived.index.equals(data.index)
    cutoff = strategy.warmup_bars(params)
    pd.testing.assert_frame_equal(
        derived.iloc[:cutoff], strategy.indicators(data.iloc[:cutoff], params))
    assert derived["fractional_z"].iloc[cutoff:].notna().all()


if __name__ == "__main__":
    _self_check()
