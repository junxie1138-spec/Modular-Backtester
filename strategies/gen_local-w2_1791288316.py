from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class SeasonalStreakParams:
    streak_weeks: int = 3
    atr_window: int = 14
    breakeven_pct: float = 0.004
    trail_atr: float = 1.0
    hold_bars: int = 2


class GeneratedStrategy(BaseStrategy[SeasonalStreakParams]):
    strategy_id = "gen_local-w2_1791288316"

    @classmethod
    def params_type(cls):
        return SeasonalStreakParams

    def warmup_bars(self, params):
        return int(max(200, params.atr_window + 1, 7 * (params.streak_weeks + 1)))

    def indicators(self, data, params):
        if params.streak_weeks < 1 or params.atr_window < 1:
            raise ValueError("Lookbacks must be positive.")
        if params.breakeven_pct <= 0 or params.trail_atr <= 0:
            raise ValueError("Stop parameters must be positive.")
        if params.hold_bars not in (1, 2):
            raise ValueError("hold_bars must be 1 or 2.")
        out = pd.DataFrame(index=data.index)
        out["ma200"] = data["close"].rolling(200, min_periods=200).mean()
        previous = data["close"].shift(1)
        out["true_range"] = pd.concat([
            data["high"] - data["low"],
            (data["high"] - previous).abs(),
            (data["low"] - previous).abs(),
        ], axis=1).max(axis=1)
        out["atr"] = out["true_range"].rolling(
            params.atr_window, min_periods=params.atr_window
        ).mean()
        out["friday"] = data.index.dayofweek == 4
        out["down_session"] = (data["close"] < data["open"]).fillna(False)
        friday_down = out.loc[out["friday"], "down_session"]
        streak = friday_down.astype(int).groupby((~friday_down).cumsum()).cumsum()
        out["friday_streak"] = streak.reindex(data.index).ffill().fillna(0).astype(int)
        out["entry"] = (
            out["friday"]
            & (out["friday_streak"] == params.streak_weeks)
            & (data["close"] > out["ma200"])
            & out["atr"].gt(0)
        ).fillna(False)
        return out

    def generate_signals(self, data, indicators, ctx: StrategyContext, params):
        df = pd.DataFrame(index=data.index)
        decisions = np.zeros(len(data), dtype=int)
        prices = data[["open", "high", "low", "close"]].to_numpy(dtype=float)
        atr = indicators["atr"].to_numpy(dtype=float)
        entries = indicators["entry"].fillna(False).to_numpy(dtype=bool)
        entry_bar = -1
        entry_price = stop = peak = np.nan
        armed = False
        for i in range(len(data)):
            opening, high, low, close = prices[i]
            if entry_bar >= 0:
                if not np.isfinite(prices[i]).all() or not np.isfinite(atr[i]) or atr[i] <= 0:
                    entry_bar = -1
                    continue
                if i == entry_bar:
                    # The prior close's decision fills at today's open.
                    entry_price = opening
                    peak = opening
                    stop = entry_price - params.trail_atr * atr[i - 1]
                    armed = False
                breached = low <= stop
                peak = max(peak, high)
                if high >= entry_price * (1.0 + params.breakeven_pct):
                    armed = True
                if armed:
                    stop = max(stop, entry_price, peak - params.trail_atr * atr[i])
                    # The high precedes the close; never assume it precedes the low.
                    breached = breached or close <= stop
                age = i - entry_bar + 1
                if breached or age >= params.hold_bars:
                    entry_bar = -1
                else:
                    decisions[i] = 1
                continue
            if entries[i] and np.isfinite(prices[i]).all() and np.isfinite(atr[i]) and atr[i] > 0:
                decisions[i] = 1
                entry_bar = i + 1
        df["signal"] = decisions
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = 1.0
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check():
    strategy = GeneratedStrategy()
    params = SeasonalStreakParams()
    index = pd.bdate_range("2020-01-01", periods=230)
    close = pd.Series(np.linspace(100.0, 130.0, len(index)), index=index)
    data = pd.DataFrame({
        "open": close - 0.1, "high": close + 0.5,
        "low": close - 0.5, "close": close, "volume": 1000000.0,
    }, index=index)
    fridays = index[index.dayofweek == 4]
    data.loc[fridays[-3:], "open"] = data.loc[fridays[-3:], "close"] + 0.1
    indicator = strategy.indicators(data, params)
    assert indicator.loc[fridays[-3:], "friday_streak"].tolist() == [1, 2, 3]
    assert bool(indicator.loc[fridays[-1], "entry"])
    bars = pd.DataFrame({
        "open": [100.0, 100.0, 100.5, 100.0],
        "high": [100.2, 100.6, 100.7, 100.2],
        "low": [99.8, 99.9, 99.95, 99.8],
        "close": [100.0, 100.5, 100.2, 100.0],
        "volume": [1000.0] * 4,
    }, index=index[:4])
    supplied = pd.DataFrame({"atr": [1.0] * 4, "entry": [True, False, False, False]}, index=bars.index)
    signal = strategy.generate_signals(bars, supplied, None, params).data
    assert signal["signal"].tolist() == [0, 1, 1, 0]
    # Breakeven activates on the first held session; its closing breach exits next open.
    bars.loc[bars.index[1], "close"] = 99.95
    assert strategy.generate_signals(bars, supplied, None, params).data["signal"].tolist() == [0, 1, 0, 0]
    # A close below the newly raised ATR trail also schedules an exit.
    bars.loc[bars.index[1], ["high", "close"]] = [102.0, 100.9]
    assert strategy.generate_signals(bars, supplied, None, params).data["signal"].tolist() == [0, 1, 0, 0]
    for end in (205, 220, 229):
        prefix = strategy.indicators(data.iloc[:end], params)
        pd.testing.assert_frame_equal(prefix, indicator.iloc[:end])
    assert signal["size"].gt(0).all()
    assert strategy.warmup_bars(params) >= 200


if __name__ == "__main__":
    _self_check()
