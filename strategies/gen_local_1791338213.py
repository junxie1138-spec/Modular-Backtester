from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class Params:
    trend_window: int = 16
    noise_window: int = 16
    atr_window: int = 14
    innovation_threshold: float = 0.04
    position_limit: float = 1.5
    trail_atr: float = 2.0
    hold_bars: int = 4


class GeneratedStrategy(BaseStrategy[Params]):
    strategy_id = "gen_local_1791338213"

    @classmethod
    def params_type(cls):
        return Params

    def warmup_bars(self, params: Params) -> int:
        return max(params.trend_window + 2, params.noise_window + 2,
                   params.atr_window + 1)

    def indicators(self, data: pd.DataFrame, params: Params) -> pd.DataFrame:
        if (params.trend_window < 3 or params.noise_window < 2
                or params.atr_window < 1 or params.hold_bars < 1
                or not np.isfinite([params.innovation_threshold,
                                   params.position_limit, params.trail_atr]).all()
                or params.innovation_threshold <= 0
                or params.position_limit <= 0 or params.trail_atr <= 0):
            raise ValueError("Invalid strategy parameters")
        out = pd.DataFrame(index=data.index)
        price = np.log(data["close"].where(data["close"] > 0))
        out["log_close"] = price
        n = params.trend_window
        weights = np.arange(n, dtype=float) - (n - 1) / 2.0
        slope = np.full(len(data), np.nan)
        if len(data) >= n:
            slope[n - 1:] = np.convolve(
                price.to_numpy(), weights[::-1] / np.sum(weights ** 2),
                mode="valid")
        out["slope"] = slope
        out["level"] = price.rolling(n, min_periods=n).mean() + out["slope"] * (n - 1) / 2.0
        out["noise"] = price.diff().rolling(
            params.noise_window, min_periods=params.noise_window).std(ddof=0).replace(0, np.nan)
        out["slope_innovation"] = out["slope"].diff() / out["noise"]
        out["relative_position"] = (price - out["level"]) / out["noise"]
        previous = data["close"].shift(1)
        out["true_range"] = pd.concat([
            data["high"] - data["low"],
            (data["high"] - previous).abs(),
            (data["low"] - previous).abs()], axis=1).max(axis=1)
        out["atr"] = out["true_range"].rolling(
            params.atr_window, min_periods=params.atr_window).mean()
        out = out.replace([np.inf, -np.inf], np.nan)
        positive = ((out["slope"] > 0)
                    & (out["slope_innovation"] > params.innovation_threshold)
                    & (out["relative_position"] > 0)
                    & (out["relative_position"] < params.position_limit))
        negative = ((out["slope"] < 0)
                    & (out["slope_innovation"] < -params.innovation_threshold)
                    & (out["relative_position"] < 0)
                    & (out["relative_position"] > -params.position_limit))
        out["candidate"] = np.select([positive, negative], [1, -1], default=0).astype(int)
        out["confirmed"] = out["candidate"].where(
            out["candidate"].eq(out["candidate"].shift(1)), 0).astype(int)
        return out

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame,
                         ctx: StrategyContext, params: Params) -> SignalFrame:
        signals = np.zeros(len(data), dtype=int)
        position, age = 0, 0
        extreme = stop = np.nan
        closes = data["close"].to_numpy(dtype=float)
        atrs = indicators["atr"].to_numpy(dtype=float)
        entries = indicators["confirmed"].to_numpy(dtype=int)
        for i in range(len(data)):
            close, atr = closes[i], atrs[i]
            if not np.isfinite(close) or not np.isfinite(atr) or close <= 0 or atr <= 0:
                position, age = 0, 0
                extreme = stop = np.nan
            elif position:
                age += 1
                if position == 1:
                    extreme = close if np.isnan(extreme) else max(extreme, close)
                    proposed = extreme - params.trail_atr * atr
                    stop = proposed if np.isnan(stop) else max(stop, proposed)
                    hit = close <= stop
                else:
                    extreme = close if np.isnan(extreme) else min(extreme, close)
                    proposed = extreme + params.trail_atr * atr
                    stop = proposed if np.isnan(stop) else min(stop, proposed)
                    hit = close >= stop
                if hit or age >= params.hold_bars:
                    position, age = 0, 0
                    extreme = stop = np.nan
            elif entries[i]:
                position = int(entries[i])
                age = 0
                # Entry fills next bar; extrema start with that bar's close.
                extreme = stop = np.nan
            signals[i] = position
        df = pd.DataFrame({"signal": signals, "size": 1.0}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check():
    params = Params(trail_atr=2.0, hold_bars=5, position_limit=10.0)
    index = pd.date_range("2020-01-01", periods=7, freq="D")
    for direction in (1, -1):
        closes = np.array([100, 100, 103, 102, 100.5, 100, 100])
        if direction == -1:
            closes = 200 - closes
        data = pd.DataFrame({"close": closes}, index=index)
        derived = pd.DataFrame({
            "atr": [1, 1, 1, 3, 3, 3, 3],
            "confirmed": [direction, 0, 0, 0, 0, 0, 0]}, index=index)
        result = GeneratedStrategy.generate_signals(None, data, derived, None, params).data
        assert result["signal"].tolist() == [0, direction, direction, direction, direction, 0, 0]
        assert result["size"].gt(0).all()
    flat = pd.DataFrame({"close": np.full(7, 100.0)}, index=index)
    derived = pd.DataFrame({"atr": np.ones(7), "confirmed": [1, 0, 0, 0, 0, 0, 0]}, index=index)
    timed = GeneratedStrategy.generate_signals(None, flat, derived, None, Params()).data
    assert timed["signal"].tolist() == [0, 1, 1, 1, 1, 0, 0]
    prices = np.exp(np.linspace(4.5, 4.6, 80) + 0.00005 * np.arange(80) ** 2)
    data = pd.DataFrame({"close": prices, "high": prices + 1, "low": prices - 1},
                        index=pd.date_range("2020-01-01", periods=80))
    derived = GeneratedStrategy.indicators(None, data, params)
    assert derived.index.equals(data.index)
    assert derived["confirmed"].ne(0).any()
    assert (derived.loc[derived["confirmed"].ne(0), "candidate"]
            == derived["candidate"].shift(1).loc[derived["confirmed"].ne(0)]).all()
    assert derived["confirmed"].iloc[:GeneratedStrategy.warmup_bars(None, params) - 1].eq(0).all()


if __name__ == "__main__":
    _self_check()
