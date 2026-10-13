from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class GeneratedParams:
    streak: int = 3
    atr_period: int = 14
    breakeven_pct: float = 0.01
    trail_atr: float = 2.0
    max_hold: int = 10


class GeneratedStrategy(BaseStrategy[GeneratedParams]):
    strategy_id = "gen_local_1791267596"

    @classmethod
    def params_type(cls):
        return GeneratedParams

    def warmup_bars(self, params):
        return int(max(200, params.atr_period + 1, params.streak + 2))

    def indicators(self, data, params):
        if (params.streak < 3 or params.atr_period < 1 or params.max_hold < 1
                or not np.isfinite(params.breakeven_pct) or params.breakeven_pct <= 0
                or not np.isfinite(params.trail_atr) or params.trail_atr <= 0):
            raise ValueError("Invalid streak, ATR, holding period or stop parameters")
        out = pd.DataFrame(index=data.index)
        close = data["close"]
        previous = close.shift(1)
        out["ma200"] = close.rolling(200, min_periods=200).mean()
        out["true_range"] = pd.concat(
            [data["high"] - data["low"],
             (data["high"] - previous).abs(),
             (data["low"] - previous).abs()], axis=1
        ).max(axis=1).where(previous.notna())
        out["atr"] = out["true_range"].rolling(
            params.atr_period, min_periods=params.atr_period
        ).mean()
        out["body"] = (close - data["open"]).abs()
        red = close.lt(data["open"])
        green = close.gt(data["open"])
        out["down_streak"] = red.groupby((~red).cumsum()).cumsum().clip(upper=params.streak)
        out["up_streak"] = green.groupby((~green).cumsum()).cumsum().clip(upper=params.streak)
        out["exhausted"] = (
            out["body"].lt(out["body"].shift(1))
            & out["body"].shift(1).lt(out["body"].shift(2))
        )
        ready = np.isfinite(out["ma200"]) & np.isfinite(out["atr"]) & out["atr"].gt(0)
        long_entry = (
            out["down_streak"].eq(params.streak)
            & out["down_streak"].shift(1).lt(params.streak)
            & close.gt(out["ma200"])
        )
        short_entry = (
            out["up_streak"].eq(params.streak)
            & out["up_streak"].shift(1).lt(params.streak)
            & close.lt(out["ma200"])
        )
        out["entry"] = np.select(
            [ready & out["exhausted"] & long_entry,
             ready & out["exhausted"] & short_entry], [1, -1], default=0
        )
        return out

    def generate_signals(self, data, indicators, ctx, params):
        raw = np.zeros(len(data), dtype=int)
        side = 0
        entry = np.nan
        entry_atr = np.nan
        stop = -np.inf
        peak = -np.inf
        armed = False
        age = 0
        for i in range(len(data)):
            close = float(data["close"].iloc[i])
            atr = float(indicators["atr"].iloc[i])
            if side:
                if np.isnan(entry):
                    # Initialize only when the prior close's order reaches its fill bar.
                    entry = float(data["open"].iloc[i])
                    if not np.isfinite(entry) or entry <= 0:
                        side = 0
                        continue
                    peak = side * entry
                    stop = peak - params.trail_atr * entry_atr
                age += 1
                if not np.isfinite(close) or close <= 0 or not np.isfinite(atr) or atr <= 0:
                    side = 0
                    continue
                signed_close = side * close
                peak = max(peak, signed_close)
                if signed_close - side * entry >= entry * params.breakeven_pct:
                    armed = True
                if armed:
                    # Signed stops increase only; the corresponding short buy-stop decreases.
                    stop = max(stop, side * entry, peak - params.trail_atr * atr)
                # Close-observed exits execute on the following bar, including gap-through stops.
                if signed_close <= stop or age >= params.max_hold:
                    side = 0
                raw[i] = side
            else:
                candidate = int(indicators["entry"].iloc[i])
                if (candidate and np.isfinite(close) and close > 0
                        and np.isfinite(atr) and atr > 0):
                    side = candidate
                    entry = np.nan
                    entry_atr = atr
                    stop = -np.inf
                    peak = -np.inf
                    armed = False
                    age = 0
                raw[i] = side
        df = pd.DataFrame(index=data.index)
        df["signal"] = raw
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = 1.0
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check():
    strategy = GeneratedStrategy()
    params = GeneratedParams(trail_atr=1.0)
    index = pd.date_range("2020-01-01", periods=5)
    ctx = StrategyContext(symbol="SPY", timeframe="1d", warmup_bars=200)
    for side in (1, -1):
        closes = np.array([100.0, 101.5, 102.0, 100.8, 100.8])
        if side == -1:
            closes = 200.0 - closes
        data = pd.DataFrame(
            {"open": 100.0, "high": closes + 1, "low": closes - 1,
             "close": closes, "volume": 1000.0}, index=index
        )
        indicators = pd.DataFrame(
            {"atr": 1.0, "entry": [side, 0, 0, 0, 0]}, index=index
        )
        signals = strategy.generate_signals(data, indicators, ctx, params).data
        assert signals["signal"].tolist() == [0, side, side, side, 0]
        assert signals["size"].gt(0).all()


if __name__ == "__main__":
    _self_check()
