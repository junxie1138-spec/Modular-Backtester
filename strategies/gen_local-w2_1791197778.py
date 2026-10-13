from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class StreakParams:
    record_window: int = 40
    min_streak: int = 4
    atr_period: int = 14
    breakeven_pct: float = 0.015
    trail_atr: float = 3.0
    hold_bars: int = 18


class GeneratedStrategy(BaseStrategy[StreakParams]):
    strategy_id = "gen_local-w2_1791197778"

    @classmethod
    def params_type(cls):
        return StreakParams

    def warmup_bars(self, params):
        if min(params.record_window, params.min_streak, params.atr_period, params.hold_bars) < 1:
            raise ValueError("Lookbacks and holding period must be positive.")
        if not (np.isfinite(params.breakeven_pct) and params.breakeven_pct > 0):
            raise ValueError("breakeven_pct must be positive and finite.")
        if not (np.isfinite(params.trail_atr) and params.trail_atr > 0):
            raise ValueError("trail_atr must be positive and finite.")
        return int(max(200, params.record_window + 2, params.atr_period + 1))

    def indicators(self, data, params):
        self.warmup_bars(params)
        out = pd.DataFrame(index=data.index)
        close = data["close"].replace([np.inf, -np.inf], np.nan)
        out["delta"] = close.diff()
        up = out["delta"].gt(0)
        down = out["delta"].lt(0)
        out["up_streak"] = up.astype(int).groupby((~up).cumsum()).cumsum()
        out["down_streak"] = down.astype(int).groupby((~down).cumsum()).cumsum()
        out["up_record"] = out["up_streak"].shift(1).rolling(params.record_window).max()
        out["down_record"] = out["down_streak"].shift(1).rolling(params.record_window).max()
        out["ma200"] = close.rolling(200).mean()
        out["true_range"] = pd.concat([
            data["high"] - data["low"],
            (data["high"] - close.shift(1)).abs(),
            (data["low"] - close.shift(1)).abs(),
        ], axis=1).max(axis=1).replace([np.inf, -np.inf], np.nan)
        out["atr"] = out["true_range"].rolling(params.atr_period).mean()
        ready = out[["ma200", "atr", "up_record", "down_record"]].notna().all(axis=1)
        ready &= close.gt(0) & out["atr"].gt(0)
        out["long_break"] = ready & close.gt(out["ma200"]) & out["up_streak"].ge(params.min_streak) & out["up_streak"].gt(out["up_record"])
        out["short_break"] = ready & close.lt(out["ma200"]) & out["down_streak"].ge(params.min_streak) & out["down_streak"].gt(out["down_record"])
        long_pulse = out["long_break"] & ~out["long_break"].shift(1, fill_value=False)
        short_pulse = out["short_break"] & ~out["short_break"].shift(1, fill_value=False)
        out["entry"] = long_pulse.astype(int) - short_pulse.astype(int)
        return out

    def generate_signals(self, data, indicators, ctx, params):
        self.warmup_bars(params)
        raw = np.zeros(len(data), dtype=int)
        opens = data["open"].to_numpy(dtype=float)
        closes = data["close"].to_numpy(dtype=float)
        atr = indicators["atr"].to_numpy(dtype=float)
        ma = indicators["ma200"].to_numpy(dtype=float)
        entries = indicators["entry"].fillna(0).to_numpy(dtype=int)
        position = 0
        entry_price = 0.0
        entry_bar = 0
        stop = 0.0
        armed = False
        for i in range(len(data)):
            # The previous close's decision fills at this bar's open.
            if position == 0 and i > 0 and raw[i - 1] != 0:
                if not (np.isfinite(opens[i]) and opens[i] > 0 and np.isfinite(atr[i - 1]) and atr[i - 1] > 0):
                    continue
                position = int(raw[i - 1])
                entry_price = opens[i]
                entry_bar = i
                stop = -params.trail_atr * atr[i - 1]
                armed = False
            valid = np.isfinite(closes[i]) and closes[i] > 0 and np.isfinite(atr[i]) and atr[i] > 0 and np.isfinite(ma[i])
            if position:
                if not valid:
                    position = 0
                    continue
                # Stops use directional dollars of profit; max tightens both sides.
                profit = position * (closes[i] - entry_price)
                if profit >= entry_price * params.breakeven_pct:
                    armed = True
                if armed:
                    stop = max(stop, 0.0, profit - params.trail_atr * atr[i])
                regime_lost = position * (closes[i] - ma[i]) <= 0
                if profit <= stop or regime_lost or i - entry_bar + 1 >= params.hold_bars:
                    position = 0
                    continue
                raw[i] = position
            elif valid:
                candidate = int(entries[i])
                if candidate * (closes[i] - ma[i]) > 0:
                    raw[i] = candidate
        df = pd.DataFrame({"signal": raw, "size": 1.0}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check():
    strategy = GeneratedStrategy()
    params = StreakParams()
    index = pd.date_range("2020-01-01", periods=7)
    ctx = StrategyContext(symbol="SPY", timeframe="1d", warmup_bars=200)
    for direction in (1, -1):
        close = 100.0 + direction * np.array([0, 0, 2, 5, 3, 1, 1], dtype=float)
        data = pd.DataFrame({"open": 100.0, "high": close + 1, "low": close - 1, "close": close, "volume": 1000.0}, index=index)
        ind = pd.DataFrame({"atr": [1, 1, 1, 1, 10, 10, 10], "ma200": 100.0 - direction, "entry": [direction, 0, 0, 0, 0, 0, 0]}, index=index)
        result = strategy.generate_signals(data, ind, ctx, params).data
        assert result["signal"].tolist() == [0, direction, direction, direction, direction, direction, 0]
        assert result["size"].gt(0).all()
        short_hold = StreakParams(hold_bars=3)
        timed = strategy.generate_signals(data, ind, ctx, short_hold).data
        assert timed["signal"].tolist() == [0, direction, direction, direction, 0, 0, 0]
    index = pd.date_range("2021-01-01", periods=260)
    close = 100.0 + np.arange(260) * 0.01 + (np.arange(260) % 2) * 0.1
    close[220:225] = close[219] + np.arange(1, 6) * 0.3
    data = pd.DataFrame({"open": close, "high": close + 1, "low": close - 1, "close": close, "volume": 1000.0}, index=index)
    ind = strategy.indicators(data, params)
    assert not ind["entry"].iloc[:199].any()
    assert ind["entry"].iloc[222] == 1
    full = strategy.generate_signals(data, ind, ctx, params).data
    prefix = data.iloc[:240]
    partial = strategy.generate_signals(prefix, strategy.indicators(prefix, params), ctx, params).data
    pd.testing.assert_frame_equal(full.iloc[:240], partial)


if __name__ == "__main__":
    _self_check()
