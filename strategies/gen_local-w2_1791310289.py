from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class OrdinalEscapeParams:
    complexity_window: int = 30
    min_complexity: float = 0.65
    min_surprise: float = 3.6
    atr_window: int = 14
    trail_atr: float = 1.2


class GeneratedStrategy(BaseStrategy[OrdinalEscapeParams]):
    strategy_id = "gen_local-w2_1791310289"

    @classmethod
    def params_type(cls):
        return OrdinalEscapeParams

    def warmup_bars(self, params):
        return max(200, params.complexity_window + 5, params.atr_window + 1)

    def indicators(self, data, params):
        if params.complexity_window < 2 or params.atr_window < 1:
            raise ValueError("Lookbacks must be positive; complexity_window must be >= 2.")
        if not (0.0 <= params.min_complexity <= 1.0):
            raise ValueError("min_complexity must be in [0, 1].")
        if not np.isfinite(params.min_surprise) or params.min_surprise < 0:
            raise ValueError("min_surprise must be finite and nonnegative.")
        if not np.isfinite(params.trail_atr) or params.trail_atr <= 0:
            raise ValueError("trail_atr must be finite and positive.")
        out = pd.DataFrame(index=data.index)
        close = data["close"]
        out["change"] = close.diff()
        out["older_change"] = out["change"].shift(2)
        out["middle_change"] = out["change"].shift(1)
        a, b, c = out["older_change"], out["middle_change"], out["change"]
        out["pattern"] = np.select(
            [(a < b) & (b < c), (a < c) & (c < b),
             (b < a) & (a < c), (b < c) & (c < a),
             (c < a) & (a < b), (c < b) & (b < a)],
            [0, 1, 2, 3, 4, 5], default=np.nan,
        )
        out["transition"] = 6 * out["pattern"].shift(1) + out["pattern"]
        window = params.complexity_window
        out["complexity"] = 0.0
        out["selected_probability"] = 0.0
        valid = out["transition"].notna()
        for code in range(36):
            counts = (out["transition"] == code).astype(float).where(valid)
            name = "transition_probability_" + str(code)
            out[name] = counts.rolling(window, min_periods=window).mean().shift(1)
            p = out[name]
            out["complexity"] += -(p * np.log(p.where(p > 0, 1.0))) / np.log(36.0)
            smoothed = (p * window + 1.0) / (window + 36.0)
            out["selected_probability"] += smoothed.where(out["transition"] == code, 0.0)
        out["surprise"] = -np.log(out["selected_probability"].where(out["selected_probability"] > 0))
        out["prior_high"] = data["high"].shift(1)
        out["prior_low"] = data["low"].shift(1)
        out["ma200"] = close.rolling(200, min_periods=200).mean()
        out["true_range"] = pd.concat(
            [data["high"] - data["low"],
             (data["high"] - close.shift(1)).abs(),
             (data["low"] - close.shift(1)).abs()], axis=1
        ).max(axis=1)
        out["atr"] = out["true_range"].rolling(params.atr_window, min_periods=params.atr_window).mean()
        return out

    def generate_signals(self, data, indicators, ctx, params):
        ind = indicators
        ready = np.isfinite(ind[["complexity", "surprise", "ma200", "atr", "prior_high", "prior_low"]]).all(axis=1)
        gate = ready & (ind["atr"] > 0) & (ind["complexity"] >= params.min_complexity) & (ind["surprise"] >= params.min_surprise)
        entries = np.select(
            [gate & (data["close"] > ind["prior_high"]) & (data["close"] > ind["ma200"]),
             gate & (data["close"] < ind["prior_low"]) & (data["close"] < ind["ma200"])],
            [1, -1], default=0,
        )
        close = data["close"].to_numpy(dtype=float)
        opening = data["open"].to_numpy(dtype=float)
        atr = ind["atr"].to_numpy(dtype=float)
        signal = np.zeros(len(data), dtype=int)
        position, age = 0, 0
        high_water, stop = np.nan, -np.inf
        for i in range(len(data)):
            if position:
                age += 1
                if not (np.isfinite(close[i]) and np.isfinite(atr[i]) and atr[i] > 0):
                    position = 0
                else:
                    if age == 1:
                        high_water = opening[i] if np.isfinite(opening[i]) else close[i]
                    high_water = max(high_water, close[i])
                    stop = max(stop, high_water - params.trail_atr * atr[i])
                    # The required high-water drawdown exit also harvests short profits.
                    if close[i] <= stop or age >= 2:
                        position = 0
            elif entries[i]:
                position = int(entries[i])
                age, high_water, stop = 0, np.nan, -np.inf
            signal[i] = position
        df = pd.DataFrame({"signal": signal, "size": 1.0}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check():
    index = pd.date_range("2020-01-01", periods=8)
    close = np.array([100., 100., 98., 100., 100., 101., 101., 101.])
    data = pd.DataFrame({"open": 100., "high": close + 1., "low": close - 1.,
                         "close": close, "volume": 1000.}, index=index)
    ind = pd.DataFrame({"complexity": 1., "surprise": 5., "ma200": 100.,
                        "atr": 1., "prior_high": 1000., "prior_low": 0.}, index=index)
    ind.loc[index[0], ["ma200", "prior_high"]] = [99., 99.]
    ind.loc[index[3], ["ma200", "prior_low"]] = [101., 101.]
    strategy = GeneratedStrategy()
    params = OrdinalEscapeParams()
    result = strategy.generate_signals(data, ind, None, params).data
    assert result["signal"].tolist() == [0, 1, 1, 0, -1, -1, 0, 0]
    assert (result["size"] > 0).all()
    empty = strategy.generate_signals(data, strategy.indicators(data, params), None, params).data
    assert (empty["signal"] == 0).all()


if __name__ == "__main__":
    _self_check()
