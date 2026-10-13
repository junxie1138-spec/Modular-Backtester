from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class ShapeParams:
    shape_bars: int = 9
    analog_bars: int = 252
    neighbors: int = 12
    percentile_bars: int = 126
    entry_percentile: float = 0.85
    atr_bars: int = 14
    stop_atr: float = 1.8
    hold_bars: int = 4


class GeneratedStrategy(BaseStrategy[ShapeParams]):
    strategy_id = "gen_local-w2_1791206916"

    @classmethod
    def params_type(cls):
        return ShapeParams

    @staticmethod
    def warmup_bars(params):
        return int(max(params.atr_bars + 1,
                       2 * params.shape_bars + params.hold_bars
                       + params.analog_bars + params.percentile_bars - 1))

    def indicators(self, data, params):
        w, history, h = params.shape_bars, params.analog_bars, params.hold_bars
        if not (3 <= w <= 10 and 3 <= h <= 5
                and 1 <= params.neighbors <= history
                and params.atr_bars >= 1 and params.percentile_bars >= 2
                and 0.0 < params.entry_percentile < 1.0
                and np.isfinite(params.stop_atr) and params.stop_atr > 0.0):
            raise ValueError("Invalid strategy parameters")
        out = pd.DataFrame(index=data.index)
        close = data["close"].astype(float)
        previous = close.shift(1)
        tr = pd.concat([data["high"] - data["low"],
                        (data["high"] - previous).abs(),
                        (data["low"] - previous).abs()], axis=1).max(axis=1)
        out["atr"] = tr.rolling(params.atr_bars, min_periods=params.atr_bars).mean()
        points = np.column_stack([close.shift(w - 1 - j).to_numpy()
                                  for j in range(w)])
        valid = np.isfinite(points).all(axis=1) & (points > 0.0).all(axis=1)
        code = np.zeros(len(data), dtype=np.uint64)
        bit = 0
        # Adjacent nodes are always linked; encode every other horizontal-visibility edge.
        for left in range(w - 2):
            obstruction = points[:, left + 1].copy()
            for right in range(left + 2, w):
                visible = valid & (obstruction < np.minimum(points[:, left], points[:, right]))
                code |= visible.astype(np.uint64) << np.uint64(bit)
                bit += 1
                obstruction = np.maximum(obstruction, points[:, right])
        out["topology_code"] = code
        out["topology_valid"] = valid
        codes = pd.Series(code, index=data.index)
        validity = pd.Series(valid, index=data.index)
        byte_counts = np.unpackbits(np.arange(256, dtype=np.uint8)[:, None], axis=1).sum(axis=1)
        byte_shifts = np.arange(8, dtype=np.uint64) * np.uint64(8)
        distances = np.full((len(data), history), np.inf)
        outcomes = np.full((len(data), history), np.nan)
        # ponytail: bounded history scan; chunk rows if much longer intraday datasets are needed.
        for column, lag in enumerate(range(w + h, w + h + history)):
            old_code = codes.shift(lag, fill_value=0).to_numpy(dtype=np.uint64)
            old_valid = validity.shift(lag, fill_value=False).to_numpy(dtype=bool)
            forward = (close.shift(lag - h) / close.shift(lag) - 1.0).to_numpy()
            usable = valid & old_valid & np.isfinite(forward)
            xor = code ^ old_code
            distance = byte_counts[((xor[:, None] >> byte_shifts) & 255).astype(np.uint8)].sum(axis=1)
            distances[:, column] = np.where(usable, distance / bit, np.inf)
            outcomes[:, column] = forward
        nearest = np.argsort(distances, axis=1, kind="stable")[:, :params.neighbors]
        matched = np.take_along_axis(outcomes, nearest, axis=1)
        complete = np.isfinite(distances).all(axis=1)
        mean_return = matched.mean(axis=1)
        mean_absolute = np.abs(matched).mean(axis=1)
        score = np.divide(mean_return, mean_absolute, out=np.zeros(len(data)),
                          where=np.isfinite(mean_absolute) & (mean_absolute > 0.0))
        out["analog_return"] = np.where(complete, mean_return, np.nan)
        out["analog_distance"] = np.where(
            complete, np.take_along_axis(distances, nearest, axis=1).mean(axis=1), np.nan)
        out["score"] = np.where(complete, score, np.nan)
        out["entry_threshold"] = out["score"].abs().shift(1).rolling(
            params.percentile_bars, min_periods=params.percentile_bars
        ).quantile(params.entry_percentile)
        return out

    def generate_signals(self, data, indicators, ctx, params):
        df = pd.DataFrame(index=data.index)
        df["signal"] = 0
        df["size"] = 0.95
        close = data["close"].to_numpy(dtype=float)
        opening = data["open"].to_numpy(dtype=float)
        atr = indicators["atr"].to_numpy(dtype=float)
        score = indicators["score"].to_numpy(dtype=float)
        threshold = indicators["entry_threshold"].to_numpy(dtype=float)
        raw = np.zeros(len(data), dtype=int)
        position, entry_bar = 0, -1
        entry_price, fixed_atr = np.nan, np.nan
        for i in range(len(data)):
            if position:
                if i == entry_bar:
                    entry_price = opening[i]
                invalid = not (np.isfinite(entry_price) and entry_price > 0.0
                               and np.isfinite(close[i]) and close[i] > 0.0)
                stopped = (not invalid and
                           position * (close[i] - entry_price) < -params.stop_atr * fixed_atr)
                expired = i - entry_bar + 1 >= params.hold_bars
                if invalid or stopped or expired:
                    position = 0
                    continue
            elif (np.isfinite(close[i]) and close[i] > 0.0
                  and np.isfinite(atr[i]) and atr[i] > 0.0
                  and np.isfinite(score[i]) and np.isfinite(threshold[i])
                  and abs(score[i]) > threshold[i]):
                position = 1 if score[i] > 0.0 else -1
                entry_bar = i + 1
                entry_price = np.nan
                fixed_atr = atr[i]
            raw[i] = position
        df["signal"] = raw
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check():
    data = pd.DataFrame({
        "open": [100.0] * 7, "high": [101.0] * 7,
        "low": [89.0] * 7,
        "close": [100.0, 97.0, 100.0, 100.0, 100.0, 100.0, 100.0],
        "volume": [1000.0] * 7,
    }, index=pd.date_range("2020-01-01", periods=7))
    indicators = pd.DataFrame({
        "atr": [1.0] * 7, "score": [1.0] + [0.0] * 6,
        "entry_threshold": [0.5] * 7,
    }, index=data.index)
    strategy = GeneratedStrategy()
    signals = strategy.generate_signals(data, indicators, None, ShapeParams()).data
    assert signals["signal"].tolist() == [0, 1, 0, 0, 0, 0, 0]
    data["close"] = 100.0
    signals = strategy.generate_signals(data, indicators, None, ShapeParams()).data
    assert signals["signal"].tolist() == [0, 1, 1, 1, 1, 0, 0]
    indicators["score"] = [-1.0] + [0.0] * 6
    data.loc[data.index[1], "close"] = 103.0
    signals = strategy.generate_signals(data, indicators, None, ShapeParams()).data
    assert signals["signal"].tolist() == [0, -1, 0, 0, 0, 0, 0]


if __name__ == "__main__":
    _self_check()
