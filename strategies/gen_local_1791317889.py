from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class Params:
    peak_window: int = 60
    shelf_window: int = 5
    compression: float = 0.006
    min_depth: float = 0.025
    max_depth: float = 0.10
    atr_window: int = 14
    breakeven_pct: float = 0.01
    trail_atr: float = 2.5
    max_hold: int = 10


class GeneratedStrategy(BaseStrategy[Params]):
    strategy_id = "gen_local_1791317889"

    @classmethod
    def params_type(cls):
        return Params

    def warmup_bars(self, params):
        return int(max(200, params.peak_window + params.shelf_window,
                       params.atr_window + 1))

    def indicators(self, data, params):
        if (params.peak_window < 2 or params.shelf_window < 2
                or params.atr_window < 1 or params.max_hold < 1
                or not 0 < params.compression < 1
                or not 0 < params.min_depth < params.max_depth < 1
                or not 0 < params.breakeven_pct < 1
                or not np.isfinite(params.trail_atr) or params.trail_atr <= 0):
            raise ValueError("Invalid strategy parameters")
        out = pd.DataFrame(index=data.index)
        close = data["close"]
        out["ma200"] = close.rolling(200, min_periods=200).mean()
        out["peak"] = data["high"].rolling(
            params.peak_window, min_periods=params.peak_window).max()
        out["depth"] = 1.0 - close / out["peak"].where(out["peak"] > 0)
        prior_depth = out["depth"].shift(1)
        shelf = prior_depth.rolling(params.shelf_window,
                                    min_periods=params.shelf_window)
        out["shelf_min"] = shelf.min()
        out["shelf_max"] = shelf.max()
        out["shelf_mean"] = shelf.mean()
        out["shelf_width"] = out["shelf_max"] - out["shelf_min"]
        previous = close.shift(1)
        out["true_range"] = pd.concat([
            data["high"] - data["low"],
            (data["high"] - previous).abs(),
            (data["low"] - previous).abs(),
        ], axis=1).max(axis=1).where(previous.notna())
        out["atr"] = out["true_range"].rolling(
            params.atr_window, min_periods=params.atr_window).mean()
        return out.replace([np.inf, -np.inf], np.nan)

    def generate_signals(self, data, indicators, ctx, params):
        ready = indicators[[
            "ma200", "depth", "shelf_min", "shelf_max",
            "shelf_mean", "shelf_width", "atr",
        ]].notna().all(axis=1)
        shelf = (ready
                 & (indicators["shelf_width"] <= params.compression)
                 & indicators["shelf_mean"].between(
                     params.min_depth, params.max_depth)
                 & (indicators["atr"] > 0))
        long_entry = (shelf & (data["close"] > indicators["ma200"])
                      & (indicators["depth"] < indicators["shelf_min"]))
        short_entry = (shelf & (data["close"] < indicators["ma200"])
                       & (indicators["depth"] > indicators["shelf_max"]))
        candidates = np.select([long_entry, short_entry], [1, -1], default=0)
        raw = np.zeros(len(data), dtype=int)
        opens = data["open"].to_numpy(dtype=float)
        highs = data["high"].to_numpy(dtype=float)
        lows = data["low"].to_numpy(dtype=float)
        closes = data["close"].to_numpy(dtype=float)
        atr = indicators["atr"].to_numpy(dtype=float)
        side = 0
        pending = False
        entry = stop = best = 0.0
        armed = False
        age = 0
        for i in range(len(data)):
            if pending:
                # The previous close's order fills at this bar's open.
                entry = opens[i]
                if not np.isfinite(entry) or entry <= 0:
                    side = 0
                else:
                    best = side * entry
                    stop = best - params.trail_atr * atr[i - 1]
                    armed = False
                    age = 0
                pending = False
            if side:
                age += 1
                adverse = lows[i] if side == 1 else -highs[i]
                valid_bar = np.isfinite(
                    [opens[i], highs[i], lows[i], closes[i], atr[i]]
                ).all() and atr[i] > 0
                # Test the old stop before adjusting it at today's close.
                if not valid_bar or adverse <= stop or age >= params.max_hold:
                    side = 0
                else:
                    favorable = side * closes[i]
                    best = max(best, favorable)
                    if side * (closes[i] - entry) / entry >= params.breakeven_pct:
                        armed = True
                    if armed:
                        # Signed stop only rises; short buy-stop prices fall.
                        stop = max(stop, side * entry,
                                   best - params.trail_atr * atr[i])
            elif candidates[i] and np.isfinite(
                    [opens[i], highs[i], lows[i], closes[i]]
            ).all() and closes[i] > 0:
                side = int(candidates[i])
                pending = True
            raw[i] = side
        df = pd.DataFrame({"signal": raw, "size": 0.95}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check():
    strategy = GeneratedStrategy()
    params = Params(trail_atr=2.0)
    index = pd.date_range("2020-01-01", periods=5)
    ctx = StrategyContext(symbol="SPY", timeframe="1d", warmup_bars=200)
    for side in (1, -1):
        close = [100.0, 100.0 + side * 2, 100.0, 100.0, 100.0]
        data = pd.DataFrame({
            "open": 100.0,
            "high": [100.5, 102.5 if side == 1 else 100.5, 101.0, 100.5, 100.5],
            "low": [99.5, 99.5 if side == 1 else 97.5, 99.0, 99.5, 99.5],
            "close": close,
            "volume": 1000.0,
        }, index=index)
        ind = pd.DataFrame({
            "ma200": 90.0 if side == 1 else 110.0,
            "depth": [0.02 if side == 1 else 0.04, 0.032, 0.032, 0.032, 0.032],
            "shelf_min": 0.03, "shelf_max": 0.035,
            "shelf_mean": 0.032, "shelf_width": 0.005, "atr": 1.0,
        }, index=index)
        result = strategy.generate_signals(data, ind, ctx, params).data
        assert result["signal"].tolist() == [0, side, side, 0, 0]
        assert (result["size"] > 0).all()
    flat = pd.DataFrame({
        "open": 100.0, "high": 101.0, "low": 99.0,
        "close": 100.0, "volume": 1000.0,
    }, index=pd.date_range("2020-01-01", periods=220))
    ind = strategy.indicators(flat, Params())
    assert strategy.generate_signals(flat, ind, ctx, Params()).data["signal"].eq(0).all()
    assert strategy.warmup_bars(Params()) >= 200


if __name__ == "__main__":
    _self_check()
