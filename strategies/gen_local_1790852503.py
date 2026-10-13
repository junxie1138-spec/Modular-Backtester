from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class Params:
    coil_bars: int = 4
    baseline_bars: int = 20
    atr_bars: int = 14
    compression_ratio: float = 0.70
    volume_multiple: float = 1.15
    trail_atr: float = 1.30
    hold_bars: int = 4

    def __post_init__(self):
        if min(self.coil_bars, self.baseline_bars, self.atr_bars) < 2:
            raise ValueError("Lookbacks must be at least two.")
        if not 3 <= self.hold_bars <= 5:
            raise ValueError("hold_bars must be between three and five.")
        if not 0 < self.compression_ratio < 1:
            raise ValueError("compression_ratio must be between zero and one.")
        if not self.volume_multiple > 0 or not self.trail_atr > 0:
            raise ValueError("Volume and trail multipliers must be positive.")


class GeneratedStrategy(BaseStrategy[Params]):
    strategy_id = "gen_local_1790852503"

    @classmethod
    def params_type(cls):
        return Params

    def warmup_bars(self, params: Params) -> int:
        return max(params.baseline_bars, params.coil_bars, params.atr_bars) + 2

    def indicators(self, data: pd.DataFrame, params: Params) -> pd.DataFrame:
        p = params
        out = pd.DataFrame(index=data.index)
        close = data["close"]
        prior = close.shift(1)
        out["tr"] = pd.concat([
            data["high"] - data["low"],
            (data["high"] - prior).abs(),
            (data["low"] - prior).abs(),
        ], axis=1).max(axis=1)
        out["atr"] = out["tr"].rolling(p.atr_bars).mean()
        out["coil_range"] = out["tr"].rolling(p.coil_bars).mean().shift(1)
        out["normal_range"] = out["tr"].rolling(p.baseline_bars).mean().shift(1)
        out["coil_high"] = data["high"].rolling(p.coil_bars).max().shift(1)
        out["coil_low"] = data["low"].rolling(p.coil_bars).min().shift(1)
        out["mid"] = (out["coil_high"] + out["coil_low"]) / 2
        out["volume_base"] = data["volume"].rolling(p.baseline_bars).mean().shift(1)
        out["volume_ok"] = (out["volume_base"] > 0) & (
            data["volume"] >= p.volume_multiple * out["volume_base"]
        )
        out["compressed"] = (out["normal_range"] > 0) & (
            out["coil_range"] <= p.compression_ratio * out["normal_range"]
        )
        inside = (close < out["coil_high"]) & (close > out["coil_low"])
        out["first_long"] = out["compressed"] & out["volume_ok"] & inside & (
            prior <= out["mid"]
        ) & (close > out["mid"])
        out["first_short"] = out["compressed"] & out["volume_ok"] & inside & (
            prior >= out["mid"]
        ) & (close < out["mid"])
        out["entry_long"] = out["first_long"].shift(1, fill_value=False) & (
            out["volume_ok"]
        ) & (close > prior)
        out["entry_short"] = out["first_short"].shift(1, fill_value=False) & (
            out["volume_ok"]
        ) & (close < prior)
        return out.replace([np.inf, -np.inf], np.nan)

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame,
                         ctx: StrategyContext, params: Params) -> SignalFrame:
        n = len(data)
        signals = np.zeros(n, dtype=int)
        closes = data["close"].to_numpy(dtype=float)
        atrs = indicators["atr"].to_numpy(dtype=float)
        longs = indicators["entry_long"].fillna(False).to_numpy(dtype=bool)
        shorts = indicators["entry_short"].fillna(False).to_numpy(dtype=bool)
        position = age = 0
        extreme = stop = np.nan
        for i in range(n):
            close, atr = closes[i], atrs[i]
            if not np.isfinite(close) or not np.isfinite(atr) or atr <= 0:
                position = age = 0
                extreme = stop = np.nan
                continue
            if position:
                # Entry fills today after yesterday's decision; start tracking here.
                age += 1
                if position == 1:
                    extreme = close if age == 1 else max(extreme, close)
                    candidate = extreme - params.trail_atr * atr
                    stop = candidate if age == 1 else max(stop, candidate)
                    stopped = close <= stop
                else:
                    extreme = close if age == 1 else min(extreme, close)
                    candidate = extreme + params.trail_atr * atr
                    stop = candidate if age == 1 else min(stop, candidate)
                    stopped = close >= stop
                if stopped or age >= params.hold_bars:
                    position = age = 0
                    extreme = stop = np.nan
            elif longs[i] or shorts[i]:
                position = 1 if longs[i] else -1
                age = 0
            signals[i] = position
        df = pd.DataFrame({"signal": signals, "size": 1.0}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check():
    idx = pd.date_range("2020-01-01", periods=10)
    data = pd.DataFrame({"close": [100, 101, 103, 101, 100, 100, 99, 101, 100, 100]}, index=idx)
    ind = pd.DataFrame({"atr": 1.0, "entry_long": False, "entry_short": False}, index=idx)
    ind.loc[idx[0], "entry_long"] = True
    ind.loc[idx[5], "entry_short"] = True
    strategy = GeneratedStrategy()
    result = strategy.generate_signals(data, ind, None, Params(trail_atr=1.0)).data
    assert result["signal"].tolist() == [0, 1, 1, 1, 0, 0, -1, -1, 0, 0]
    ind["entry_short"] = False
    data["close"] = 100.0
    result = strategy.generate_signals(data, ind, None, Params()).data
    assert result["signal"].tolist() == [0, 1, 1, 1, 1, 0, 0, 0, 0, 0]
    idx = pd.date_range("2020-02-01", periods=26)
    bars = pd.DataFrame({"open": 100.0, "high": 101.0, "low": 99.0,
                         "close": 100.0, "volume": 100.0}, index=idx)
    bars.loc[idx[20:24], ["high", "low"]] = [100.2, 99.8]
    bars.loc[idx[24], ["high", "low", "close", "volume"]] = [100.3, 99.9, 100.1, 200.0]
    bars.loc[idx[25], ["high", "low", "close", "volume"]] = [100.3, 100.0, 100.15, 200.0]
    derived = strategy.indicators(bars, Params())
    assert not derived["entry_long"].iloc[24]
    assert derived["entry_long"].iloc[25]
    bars.loc[idx[25], "volume"] = 100.0
    assert not strategy.indicators(bars, Params())["entry_long"].iloc[25]


if __name__ == "__main__":
    _self_check()
