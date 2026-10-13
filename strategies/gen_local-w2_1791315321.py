from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class Params:
    atr_period: int = 14
    gap_atr: float = 0.25
    breakeven_pct: float = 0.01
    trail_atr: float = 1.5
    initial_stop_atr: float = 2.0
    hold_days: int = 4


class GeneratedStrategy(BaseStrategy[Params]):
    strategy_id = "gen_local-w2_1791315321"

    @classmethod
    def params_type(cls):
        return Params

    def warmup_bars(self, params):
        return int(params.atr_period) + 1

    def indicators(self, data, params):
        if (params.atr_period < 1 or params.gap_atr <= 0
                or params.breakeven_pct <= 0 or params.trail_atr <= 0
                or params.initial_stop_atr <= 0 or not 3 <= params.hold_days <= 5):
            raise ValueError("Invalid strategy parameters")
        previous_close = data["close"].shift(1)
        tr = pd.concat([
            data["high"] - data["low"],
            (data["high"] - previous_close).abs(),
            (data["low"] - previous_close).abs(),
        ], axis=1).max(axis=1)
        atr = tr.rolling(params.atr_period, min_periods=params.atr_period).mean()
        gap = (data["open"] - previous_close) / atr.shift(1).replace(0, np.nan)
        midpoint = (data["open"] + previous_close) / 2.0
        monday_repair = (
            (data.index.dayofweek == 0)
            & (gap <= -params.gap_atr)
            & (data["close"] > data["open"])
            & (data["close"] >= midpoint)
        ).fillna(False)
        confirmation = (
            (data.index.dayofweek == 1)
            & monday_repair.shift(1, fill_value=False)
            & (data["low"] > data["low"].shift(1))
            & (data["close"] > data["close"].shift(1))
            & (data["close"] > data["open"])
            & (data["close"] >= previous_close.shift(1))
            & atr.gt(0)
        ).fillna(False)
        return pd.DataFrame({
            "previous_close": previous_close,
            "true_range": tr,
            "atr": atr,
            "gap_atr": gap,
            "repair_midpoint": midpoint,
            "monday_repair": monday_repair,
            "confirmed_entry": confirmation,
        }, index=data.index)

    def generate_signals(self, data, indicators, ctx, params):
        df = pd.DataFrame({"signal": 0, "size": 0.95}, index=data.index)
        o = data["open"].to_numpy(dtype=float)
        h = data["high"].to_numpy(dtype=float)
        l = data["low"].to_numpy(dtype=float)
        c = data["close"].to_numpy(dtype=float)
        atr = indicators["atr"].to_numpy(dtype=float)
        confirmed = indicators["confirmed_entry"].fillna(False).to_numpy(dtype=bool)
        raw = np.zeros(len(data), dtype=int)
        active = False
        pending = False
        armed = False
        entry = stop = peak = 0.0
        age = 0
        for i in range(len(data)):
            valid = (
                np.isfinite([o[i], h[i], l[i], c[i], atr[i]]).all()
                and min(o[i], h[i], l[i], c[i], atr[i]) > 0
                and l[i] <= min(o[i], c[i])
                and h[i] >= max(o[i], c[i])
            )
            if pending:
                pending = False
                if valid and i > 0 and np.isfinite(atr[i - 1]) and atr[i - 1] > 0:
                    active = True
                    entry = o[i]
                    peak = entry
                    stop = entry - params.initial_stop_atr * atr[i - 1]
                    armed = False
                    age = 0
            if active:
                age += 1
                # Test the previously established stop before raising it.
                exit_now = not valid or l[i] <= stop or age >= params.hold_days
                if not exit_now:
                    peak = max(peak, h[i])
                    armed = armed or peak >= entry * (1.0 + params.breakeven_pct)
                    if armed:
                        stop = max(stop, entry, peak - params.trail_atr * atr[i])
                    exit_now = c[i] <= stop
                if exit_now:
                    active = False
                else:
                    raw[i] = 1
            elif valid and i >= self.warmup_bars(params) - 1 and confirmed[i]:
                # This decision fills at the next bar's open.
                pending = True
                raw[i] = 1
        df["signal"] = raw
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check():
    index = pd.bdate_range("2024-01-02", periods=11)
    data = pd.DataFrame({
        "open": [100, 100, 100, 100, 99, 99.9, 100.6, 101, 102, 101, 101],
        "high": [100.5, 100.5, 100.5, 100.5, 100.1, 100.8, 101, 102.4, 102.2, 102, 102],
        "low": [99.5, 99.5, 99.5, 99.5, 98.8, 99.2, 100.3, 100.8, 100.5, 100, 100],
        "close": [100, 100, 100, 100, 99.8, 100.5, 100.8, 102, 101, 101, 101],
        "volume": [1000000] * 11,
    }, index=index)
    strategy = GeneratedStrategy()
    params = Params(atr_period=2, gap_atr=0.1)
    ctx = StrategyContext("SPY", "1d", strategy.warmup_bars(params))
    result = strategy.generate_signals(data, strategy.indicators(data, params), ctx, params).data
    assert result["signal"].tolist() == [0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0]
    params = Params(atr_period=2, gap_atr=0.1, breakeven_pct=1.0, initial_stop_atr=10.0)
    result = strategy.generate_signals(data, strategy.indicators(data, params), ctx, params).data
    assert result["signal"].tolist() == [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0]
    prefix = data.iloc[:8]
    shorter = strategy.generate_signals(prefix, strategy.indicators(prefix, params), ctx, params).data
    assert shorter.equals(result.iloc[:8])
    rejected = data.copy()
    rejected.iloc[5, rejected.columns.get_loc("close")] = 99.7
    result = strategy.generate_signals(rejected, strategy.indicators(rejected, params), ctx, params).data
    assert not result["signal"].any()


if __name__ == "__main__":
    _self_check()
