from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class GapMidpointParams:
    atr_period: int = 14
    gap_atr: float = 0.25
    breakeven_pct: float = 0.004
    trail_atr: float = 0.8


class GeneratedStrategy(BaseStrategy[GapMidpointParams]):
    strategy_id = "gen_local_1790924625"

    @classmethod
    def params_type(cls):
        return GapMidpointParams

    def warmup_bars(self, params):
        return int(params.atr_period) + 3

    def indicators(self, data, params):
        if params.atr_period < 2 or min(params.gap_atr, params.breakeven_pct, params.trail_atr) <= 0:
            raise ValueError("Require atr_period >= 2 and positive thresholds.")
        prior_close = data["close"].shift(1)
        tr = pd.concat([
            data["high"] - data["low"],
            (data["high"] - prior_close).abs(),
            (data["low"] - prior_close).abs(),
        ], axis=1).max(axis=1)
        atr = tr.rolling(params.atr_period, min_periods=params.atr_period).mean()
        midpoint = (data["high"].shift(1) + data["low"].shift(1)) / 2.0
        gap = (data["open"] - prior_close) / atr.shift(1).replace(0, np.nan)
        long_setup = (
            (gap <= -params.gap_atr) & (prior_close > midpoint)
            & (data["open"] < midpoint) & (data["close"] > midpoint)
            & (data["close"] > data["open"])
        )
        short_setup = (
            (gap >= params.gap_atr) & (prior_close < midpoint)
            & (data["open"] > midpoint) & (data["close"] < midpoint)
            & (data["close"] < data["open"])
        )
        return pd.DataFrame({
            "tr": tr, "atr": atr, "prior_close": prior_close,
            "midpoint": midpoint, "gap_atr": gap,
            "long_setup": long_setup.fillna(False),
            "short_setup": short_setup.fillna(False),
            "long_entry": (
                long_setup.shift(1, fill_value=False)
                & (data["close"] > midpoint.shift(1))
                & (data["close"] >= prior_close)
            ).fillna(False),
            "short_entry": (
                short_setup.shift(1, fill_value=False)
                & (data["close"] < midpoint.shift(1))
                & (data["close"] <= prior_close)
            ).fillna(False),
        }, index=data.index)

    def generate_signals(self, data, indicators, ctx, params):
        decisions = np.zeros(len(data), dtype=int)
        side, age = 0, 0
        entry = None
        stop, best, armed = 0.0, 0.0, False
        for i in range(len(data)):
            atr = float(indicators["atr"].iloc[i])
            close = float(data["close"].iloc[i])
            if side:
                if entry is None:
                    # The previous close's decision fills at this bar's open.
                    entry = float(data["open"].iloc[i])
                    entry_atr = float(indicators["atr"].iloc[i - 1])
                    if not np.isfinite(entry) or entry <= 0 or not np.isfinite(entry_atr):
                        side = 0
                        entry = None
                        continue
                    best = side * entry
                    stop = best - params.trail_atr * entry_atr
                    age, armed = 0, False
                age += 1
                if not np.isfinite(close) or not np.isfinite(atr) or atr <= 0:
                    side = 0
                    entry = None
                    continue
                # Stops and activation use closes; every exit fills next open.
                signed_close = side * close
                best = max(best, signed_close)
                if best - side * entry >= entry * params.breakeven_pct:
                    armed = True
                if armed:
                    # Signed stops rise monotonically for both trade directions.
                    stop = max(stop, side * entry, best - params.trail_atr * atr)
                if signed_close <= stop or age >= 2:
                    side = 0
                    entry = None
                decisions[i] = side
                continue
            if not np.isfinite(atr) or atr <= 0 or not np.isfinite(close):
                continue
            if bool(indicators["long_entry"].iloc[i]):
                side = 1
            elif bool(indicators["short_entry"].iloc[i]):
                side = -1
            entry = None
            decisions[i] = side
        df = pd.DataFrame({"signal": decisions, "size": 1.0}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check():
    data = pd.DataFrame({
        "open": [101, 102, 100, 103, 104, 105, 103.5],
        "high": [104, 104, 104, 104, 106, 105, 104],
        "low": [96, 100, 99, 102, 103, 103, 103],
        "close": [103, 103, 103, 103.5, 105, 103.8, 103.5],
        "volume": [1000] * 7,
    }, index=pd.date_range("2020-01-01", periods=7))
    strategy = GeneratedStrategy()
    params = GapMidpointParams(atr_period=2)
    ctx = StrategyContext(symbol="SPY", timeframe="1d", warmup_bars=5)
    indicators = strategy.indicators(data, params)
    assert not indicators["long_entry"].iloc[2]
    assert indicators["long_entry"].iloc[3]
    signal = strategy.generate_signals(data, indicators, ctx, params).data
    assert signal["signal"].tolist() == [0, 0, 0, 0, 1, 1, 0]
    stopped = data.copy()
    stopped.loc[stopped.index[4], "close"] = 90.0
    stopped.loc[stopped.index[4], "low"] = 89.0
    early = strategy.generate_signals(
        stopped, strategy.indicators(stopped, params), ctx, params
    ).data
    assert early["signal"].tolist() == [0, 0, 0, 0, 1, 0, 0]
    mirrored = data.copy()
    mirrored["open"] = 210 - data["open"]
    mirrored["close"] = 210 - data["close"]
    mirrored["high"] = 210 - data["low"]
    mirrored["low"] = 210 - data["high"]
    short = strategy.generate_signals(
        mirrored, strategy.indicators(mirrored, params), ctx, params
    ).data
    assert short["signal"].tolist() == [0, 0, 0, 0, -1, -1, 0]
    assert signal["size"].gt(0).all()


if __name__ == "__main__":
    _self_check()
