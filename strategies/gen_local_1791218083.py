from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class Params:
    channel_bars: int = 20
    atr_bars: int = 14
    fast_bars: int = 3
    edge: float = 0.20
    compression: float = 0.85
    stop_atr: float = 1.5
    hold_bars: int = 4

    def __post_init__(self):
        if not (self.channel_bars >= 2 and self.atr_bars > self.fast_bars >= 1
                and 0 < self.edge < 0.5 and 0 < self.compression <= 1
                and self.stop_atr > 0 and 3 <= self.hold_bars <= 5):
            raise ValueError("Invalid strategy parameters")


class GeneratedStrategy(BaseStrategy[Params]):
    strategy_id = "gen_local_1791218083"

    @classmethod
    def params_type(cls):
        return Params

    def warmup_bars(self, params):
        return int(max(params.channel_bars, params.atr_bars, params.fast_bars) + 1)

    def indicators(self, data, params):
        out = pd.DataFrame(index=data.index)
        close = data["close"]
        out["previous_close"] = close.shift(1)
        out["tr"] = pd.concat([
            data["high"] - data["low"],
            (data["high"] - out["previous_close"]).abs(),
            (data["low"] - out["previous_close"]).abs(),
        ], axis=1).max(axis=1)
        out["atr"] = out["tr"].rolling(params.atr_bars).mean()
        out["fast_atr"] = out["tr"].rolling(params.fast_bars).mean()
        out["channel_low"] = close.rolling(params.channel_bars).min().shift(1)
        out["channel_high"] = close.rolling(params.channel_bars).max().shift(1)
        out["channel_width"] = out["channel_high"] - out["channel_low"]
        out["position"] = (close - out["channel_low"]) / out["channel_width"].replace(0, np.nan)
        out["compression"] = out["fast_atr"] / out["atr"].replace(0, np.nan)
        out["change"] = close.diff()
        valid = (out["position"].notna() & out["compression"].notna()
                 & (out["atr"] > 0) & (out["compression"] < params.compression))
        out["direction"] = np.select([
            valid & (out["position"] <= params.edge) & (out["change"] > 0),
            valid & (out["position"] >= 1 - params.edge) & (out["change"] < 0),
        ], [1, -1], default=0).astype(int)
        out["edge_strength"] = ((out["position"] - 0.5).abs()
                                - (0.5 - params.edge)).div(params.edge).clip(0, 1)
        out["quiet_strength"] = (1 - out["compression"] / params.compression).clip(0, 1)
        out["conviction"] = ((out["edge_strength"] + out["quiet_strength"]) / 2).fillna(0)
        return out.replace([np.inf, -np.inf], np.nan)

    def generate_signals(self, data, indicators, ctx, params):
        n = len(data)
        signals = np.zeros(n, dtype=int)
        sizes = np.full(n, 0.25, dtype=float)
        opens = data["open"].to_numpy(dtype=float)
        closes = data["close"].to_numpy(dtype=float)
        atr = indicators["atr"].to_numpy(dtype=float)
        directions = indicators["direction"].fillna(0).to_numpy(dtype=int)
        conviction = indicators["conviction"].fillna(0).to_numpy(dtype=float)
        side, pending, age = 0, False, 0
        stop, exposure = np.nan, 0.25
        for i in range(n):
            if side:
                if pending:
                    # Entry fills at this open; freeze ATR from the decision bar.
                    if not np.isfinite(opens[i]) or opens[i] <= 0:
                        side, pending = 0, False
                        continue
                    stop = opens[i] - side * params.stop_atr * atr[i - 1]
                    pending = False
                age += 1
                stopped = (closes[i] < stop if side == 1 else closes[i] > stop)
                if not np.isfinite(closes[i]) or stopped or age >= params.hold_bars:
                    side = 0
                    continue
            elif (directions[i] != 0 and np.isfinite(atr[i]) and atr[i] > 0
                  and np.isfinite(closes[i]) and closes[i] > 0):
                side = int(directions[i])
                exposure = float(0.25 + 0.70 * np.clip(conviction[i], 0, 1))
                age, pending = 0, True
            signals[i], sizes[i] = side, exposure
        df = pd.DataFrame({"signal": signals, "size": sizes}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = df["size"].shift(1).fillna(0.25).astype(float)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check():
    strategy, params = GeneratedStrategy(), Params()
    index = pd.date_range("2020-01-01", periods=8)
    data = pd.DataFrame({"open": 100.0, "high": 104.0, "low": 97.0,
                         "close": [100, 100, 98, 100, 100, 100, 103, 100],
                         "volume": 1000.0}, index=index)
    ind = pd.DataFrame({"atr": [1, 100, 100, 100, 1, 100, 100, 100],
                        "direction": [1, 0, 0, 0, -1, 0, 0, 0],
                        "conviction": [0.8, 0, 0, 0, 0.2, 0, 0, 0]}, index=index)
    result = strategy.generate_signals(data, ind, None, params).data
    assert result["signal"].tolist() == [0, 1, 1, 0, 0, -1, -1, 0]
    assert np.isclose(result["size"].iloc[1], 0.81)
    assert np.isclose(result["size"].iloc[5], 0.39)
    data["close"] = 100.0
    ind["direction"] = [1, 0, 0, 0, 0, 0, 0, 0]
    result = strategy.generate_signals(data, ind, None, params).data
    assert result["signal"].tolist() == [0, 1, 1, 1, 1, 0, 0, 0]
    flat = strategy.generate_signals(data, strategy.indicators(data, params), None, params).data
    assert flat.index.equals(data.index) and flat["signal"].eq(0).all()
    assert flat["size"].gt(0).all()


if __name__ == "__main__":
    _self_check()
