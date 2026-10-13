from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class Params:
    graph_window: int = 12
    atr_window: int = 14
    entry_strength: float = 0.18
    trail_atr: float = 1.2
    max_hold_bars: int = 2


class GeneratedStrategy(BaseStrategy[Params]):
    strategy_id = "gen_local-w2_1791340611"

    @classmethod
    def params_type(cls):
        return Params

    def warmup_bars(self, params: Params) -> int:
        return int(max(params.graph_window, params.atr_window + 1))

    def indicators(self, data: pd.DataFrame, params: Params) -> pd.DataFrame:
        if not (params.graph_window >= 3 and params.atr_window >= 1
                and 0 < params.entry_strength < 1
                and np.isfinite(params.trail_atr) and params.trail_atr > 0
                and params.max_hold_bars in (1, 2)):
            raise ValueError("Invalid graph, ATR, strength, trail, or holding parameters")
        out = pd.DataFrame(index=data.index)
        close = data["close"].astype(float)
        log_close = np.log(close.where(close > 0)).replace([np.inf, -np.inf], np.nan)
        upper = np.zeros(len(data), dtype=float)
        lower = np.zeros(len(data), dtype=float)
        upper_slope = np.full(len(data), -np.inf)
        lower_slope = np.full(len(data), np.inf)
        # An endpoint sees a past vertex iff its chord clears all nearer vertices.
        for lag in range(1, params.graph_window):
            slope = ((log_close.shift(lag) - log_close) / lag).to_numpy()
            finite = np.isfinite(slope)
            upper += np.where(finite & (slope > upper_slope + 1e-12), lag, 0.0)
            lower += np.where(finite & (slope < lower_slope - 1e-12), lag, 0.0)
            upper_slope = np.fmax(upper_slope, slope)
            lower_slope = np.fmin(lower_slope, slope)
        ready = log_close.rolling(params.graph_window).count().eq(params.graph_window)
        out["log_close"] = log_close
        out["upper_span"] = pd.Series(upper, index=data.index).where(ready)
        out["lower_span"] = pd.Series(lower, index=data.index).where(ready)
        total = out["upper_span"] + out["lower_span"]
        out["graph_bias"] = (out["upper_span"] - out["lower_span"]) / total.replace(0, np.nan)
        out["displacement"] = log_close.diff(params.graph_window - 1)
        out["path_length"] = log_close.diff().abs().rolling(params.graph_window - 1).sum()
        out["efficiency"] = (out["displacement"].abs() / out["path_length"].replace(0, np.nan)).clip(0, 1)
        aligned = out["graph_bias"] * out["displacement"] > 0
        out["score"] = (out["graph_bias"] * out["efficiency"]).where(aligned, 0.0).fillna(0.0)
        previous = close.shift(1)
        out["true_range"] = pd.concat([
            data["high"] - data["low"],
            (data["high"] - previous).abs(),
            (data["low"] - previous).abs(),
        ], axis=1).max(axis=1)
        out["atr"] = out["true_range"].rolling(params.atr_window).mean()
        return out

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame,
                         ctx: StrategyContext, params: Params) -> SignalFrame:
        df = pd.DataFrame(index=data.index)
        signals = np.zeros(len(data), dtype=int)
        sizes = np.ones(len(data), dtype=float)
        position = 0
        age = 0
        high_water = low_water = stop = np.nan
        entry_size = 1.0
        closes = data["close"].to_numpy(dtype=float)
        atrs = indicators["atr"].to_numpy(dtype=float)
        scores = indicators["score"].to_numpy(dtype=float)
        for i in range(len(data)):
            close, atr, score = closes[i], atrs[i], scores[i]
            valid = np.isfinite(close) and close > 0 and np.isfinite(atr) and atr > 0
            if position:
                age += 1
                if not valid:
                    position = 0
                else:
                    high_water = max(high_water, close)
                    low_water = min(low_water, close)
                    if position == 1:
                        stop = max(stop, high_water - params.trail_atr * atr)
                        stopped = close <= stop
                    else:
                        stop = min(stop, low_water + params.trail_atr * atr)
                        stopped = close >= stop
                    if stopped or age >= params.max_hold_bars:
                        position = 0
                # An exit always gets its own flat bar before another entry.
            elif valid and np.isfinite(score) and abs(score) >= params.entry_strength:
                position = 1 if score > 0 else -1
                age = 0
                high_water = low_water = close
                stop = close - position * params.trail_atr * atr
                entry_size = float(0.25 + 0.75 * min(abs(score), 1.0))
            signals[i] = position
            if position:
                sizes[i] = entry_size
        df["signal"] = signals
        df["size"] = sizes
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = df["size"].shift(1).fillna(1.0).astype(float)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check():
    strategy = GeneratedStrategy()
    params = Params(graph_window=4, atr_window=2, entry_strength=0.18, trail_atr=1.0)
    index = pd.date_range("2020-01-01", periods=5)
    for direction, closes, expected in [
        (1, [100, 100, 100, 100, 100], [0, 0, 1, 1, 0]),
        (1, [100, 100, 97, 97, 97], [0, 0, 1, 0, 0]),
        (-1, [100, 100, 103, 103, 103], [0, 0, -1, 0, 0]),
        (1, [100, 100, 99.5, 99.5, 99.5], [0, 0, 1, 0, 0]),
    ]:
        data = pd.DataFrame({"close": closes}, index=index)
        atr = [1.0, 1.0, 1.0, 1.0, 1.0]
        if closes[2] == 99.5:
            atr[2] = 5.0
            params.trail_atr = 0.4
        else:
            params.trail_atr = 1.0
        derived = pd.DataFrame({"atr": atr, "score": [0, direction * 0.6, 0, 0, 0]}, index=index)
        result = strategy.generate_signals(data, derived, None, params).data
        assert result["signal"].tolist() == expected
        assert np.isclose(result["size"].iloc[2], 0.7)
        assert (result["size"] > 0).all()
    index = pd.date_range("2021-01-01", periods=30)
    close = 100 * np.exp(0.001 * np.arange(30) ** 2)
    data = pd.DataFrame({"open": close, "high": close + 1, "low": close - 1,
                         "close": close, "volume": 1000.0}, index=index)
    derived = strategy.indicators(data, params)
    assert derived["score"].iloc[-1] > params.entry_strength
    assert np.isfinite(derived["score"]).all()
    pd.testing.assert_frame_equal(derived.iloc[:20], strategy.indicators(data.iloc[:20], params))
    assert strategy.warmup_bars(params) >= max(params.graph_window, params.atr_window + 1)


if __name__ == "__main__":
    _self_check()
