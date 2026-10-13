from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class AnalogParams:
    peak_bars: int = 20
    shape_bars: int = 5
    archive_bars: int = 252
    neighbors: int = 7
    atr_bars: int = 14
    min_depth: float = 0.015
    max_distance: float = 0.006
    min_edge: float = 0.002
    stop_atr: float = 1.0

    def __post_init__(self):
        if min(self.peak_bars, self.shape_bars, self.atr_bars, self.neighbors) < 1:
            raise ValueError("Lookbacks and neighbor count must be positive.")
        if self.archive_bars < self.neighbors + self.shape_bars + 3:
            raise ValueError("Archive is too short.")
        values = (self.min_depth, self.max_distance, self.min_edge, self.stop_atr)
        if not all(np.isfinite(v) and v > 0 for v in values):
            raise ValueError("Thresholds must be finite and positive.")


class GeneratedStrategy(BaseStrategy[AnalogParams]):
    strategy_id = "gen_local_1791334831"

    @classmethod
    def params_type(cls):
        return AnalogParams

    def warmup_bars(self, params: AnalogParams) -> int:
        return int(max(params.peak_bars + params.shape_bars + params.archive_bars + 3,
                       params.atr_bars + 1))

    def indicators(self, data: pd.DataFrame, params: AnalogParams) -> pd.DataFrame:
        close = data["close"].where(data["close"] > 0)
        peak = close.rolling(params.peak_bars, min_periods=params.peak_bars).max()
        depth = 1.0 - close / peak
        previous = close.shift(1)
        tr = pd.concat([data["high"] - data["low"],
                        (data["high"] - previous).abs(),
                        (data["low"] - previous).abs()], axis=1).max(axis=1)
        atr = tr.rolling(params.atr_bars, min_periods=params.atr_bars).mean()
        shapes = pd.concat([depth.shift(j) for j in range(params.shape_bars)], axis=1)
        matrix = shapes.to_numpy(dtype=float)
        opens = data["open"].to_numpy(dtype=float)
        edge = np.full(len(data), np.nan)
        distance = np.full(len(data), np.nan)
        # ponytail: bounded brute-force neighbors; index the archive if intraday history becomes large.
        for t in range(self.warmup_bars(params) - 1, len(data)):
            if not np.isfinite(matrix[t]).all():
                continue
            # Historical shapes and both next-open holding sessions finish before the current shape begins.
            endpoints = np.arange(max(params.peak_bars + params.shape_bars - 2,
                                      t - params.archive_bars),
                                  t - params.shape_bars - 2)
            if len(endpoints) < params.neighbors:
                continue
            valid = (np.isfinite(matrix[endpoints]).all(axis=1)
                     & np.isfinite(opens[endpoints + 1])
                     & np.isfinite(opens[endpoints + 3])
                     & (opens[endpoints + 1] > 0)
                     & (opens[endpoints + 3] > 0))
            endpoints = endpoints[valid]
            if len(endpoints) < params.neighbors:
                continue
            errors = np.sqrt(np.mean((matrix[endpoints] - matrix[t]) ** 2, axis=1))
            nearest = np.argsort(errors, kind="stable")[:params.neighbors]
            matches = endpoints[nearest]
            edge[t] = np.median(opens[matches + 3] / opens[matches + 1] - 1.0)
            distance[t] = errors[nearest].max()
        result = pd.DataFrame({"peak": peak, "depth": depth, "previous_close": previous,
                               "true_range": tr, "atr": atr, "analog_edge": edge,
                               "analog_distance": distance}, index=data.index)
        shapes.columns = [f"depth_lag_{j}" for j in range(params.shape_bars)]
        return pd.concat([result, shapes], axis=1).replace([np.inf, -np.inf], np.nan)

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame,
                         ctx: StrategyContext, params: AnalogParams) -> SignalFrame:
        candidate = ((indicators["depth"] >= params.min_depth)
                     & (indicators["analog_edge"] >= params.min_edge)
                     & (indicators["analog_distance"] <= params.max_distance)
                     & (indicators["atr"] > 0)).fillna(False)
        candidate.iloc[:self.warmup_bars(params) - 1] = False
        confirmed = (candidate & candidate.shift(1, fill_value=False)).to_numpy(dtype=bool)
        closes = data["close"].to_numpy(dtype=float)
        atrs = indicators["atr"].to_numpy(dtype=float)
        signal = np.zeros(len(data), dtype=int)
        active = False
        age = 0
        high_water = np.nan
        stop = -np.inf
        for t in range(len(data)):
            valid = np.isfinite(closes[t]) and closes[t] > 0 and np.isfinite(atrs[t]) and atrs[t] > 0
            if active:
                age += 1
                if not valid:
                    active = False
                    continue
                # Entry fills this bar's open; the first in-trade close initializes the watermark.
                high_water = closes[t] if age == 1 else max(high_water, closes[t])
                stop = max(stop, high_water - params.stop_atr * atrs[t])
                if closes[t] <= stop or age >= 2:
                    active = False
                else:
                    signal[t] = 1
            elif valid and confirmed[t]:
                active = True
                age = 0
                high_water = np.nan
                stop = -np.inf
                signal[t] = 1
        df = pd.DataFrame({"signal": signal, "size": 1.0}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check():
    params = AnalogParams(peak_bars=2, shape_bars=2, archive_bars=8,
                         neighbors=2, atr_bars=2)
    strategy = GeneratedStrategy()
    index = pd.date_range("2020-01-01", periods=60)
    close = 100.0 + np.sin(np.arange(60) / 2.0)
    data = pd.DataFrame({"open": close, "high": close + 1.0, "low": close - 1.0,
                         "close": close, "volume": 1000.0}, index=index)
    full = strategy.indicators(data, params)
    pd.testing.assert_frame_equal(full.iloc[:40], strategy.indicators(data.iloc[:40], params))
    fixture = pd.DataFrame({"depth": 0.02, "atr": 1.0, "analog_edge": 0.0,
                            "analog_distance": 0.0}, index=index)
    t = strategy.warmup_bars(params)
    fixture.iloc[t:t + 2, fixture.columns.get_loc("analog_edge")] = 0.01
    result = strategy.generate_signals(data, fixture, None, params).data
    expected = np.zeros(len(data), dtype=int)
    expected[t + 2:t + 4] = 1
    assert np.array_equal(result["signal"].to_numpy(), expected)
    assert (result["size"] > 0).all()


if __name__ == "__main__":
    _self_check()
