from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class StreakParams:
    streak_bars: int = 3
    regime_bars: int = 20
    trend_efficiency: float = 0.35
    pressure_bars: int = 3
    pressure_threshold: float = 0.12
    atr_bars: int = 14
    breakeven_pct: float = 0.008
    trail_atr: float = 1.6
    initial_stop_atr: float = 2.0
    hold_bars: int = 4

    def __post_init__(self):
        if min(self.streak_bars, self.regime_bars, self.pressure_bars, self.atr_bars) < 1:
            raise ValueError("Lookbacks must be positive.")
        if not 3 <= self.hold_bars <= 5:
            raise ValueError("hold_bars must be between 3 and 5.")
        if not 0 < self.trend_efficiency < 1 or not 0 < self.pressure_threshold < 1:
            raise ValueError("Regime and pressure thresholds must be between 0 and 1.")
        if min(self.breakeven_pct, self.trail_atr, self.initial_stop_atr) <= 0:
            raise ValueError("Stop parameters must be positive.")


class GeneratedStrategy(BaseStrategy[StreakParams]):
    strategy_id = "gen_local-w2_1791339656"

    @classmethod
    def params_type(cls):
        return StreakParams

    def warmup_bars(self, params):
        return int(max(params.regime_bars + 1, params.atr_bars + 1,
                       params.streak_bars + 1, params.pressure_bars))

    def indicators(self, data, params):
        close = data["close"]
        delta = close.diff()
        up = delta.gt(0)
        down = delta.lt(0)
        out = pd.DataFrame(index=data.index)
        out["up_streak"] = up.groupby((~up).cumsum()).cumsum().clip(upper=params.streak_bars)
        out["down_streak"] = down.groupby((~down).cumsum()).cumsum().clip(upper=params.streak_bars)
        travel = delta.abs().rolling(params.regime_bars).sum()
        out["efficiency"] = delta.rolling(params.regime_bars).sum().abs().div(travel.replace(0, np.nan))
        span = (data["high"] - data["low"]).replace(0, np.nan)
        body = (close - data["open"]).div(span).clip(-1, 1)
        out["pressure"] = body.rolling(params.pressure_bars).mean()
        previous = close.shift(1)
        tr = pd.concat([data["high"] - data["low"],
                        (data["high"] - previous).abs(),
                        (data["low"] - previous).abs()], axis=1).max(axis=1)
        out["atr"] = tr.rolling(params.atr_bars).mean()
        return out.replace([np.inf, -np.inf], np.nan)

    def generate_signals(self, data, indicators, ctx: StrategyContext, params):
        streak = pd.Series(np.select(
            [indicators["up_streak"].ge(params.streak_bars),
             indicators["down_streak"].ge(params.streak_bars)],
            [1, -1], default=0), index=data.index)
        trending = indicators["efficiency"].ge(params.trend_efficiency)
        streak_vote = streak.where(trending, -streak)
        pressure_vote = pd.Series(np.select(
            [indicators["pressure"].ge(params.pressure_threshold),
             indicators["pressure"].le(-params.pressure_threshold)],
            [1, -1], default=0), index=data.index)
        valid = indicators[["efficiency", "pressure", "atr"]].notna().all(axis=1)
        valid &= indicators["atr"].gt(0)
        candidate = streak_vote.where(valid & streak_vote.eq(pressure_vote), 0)
        opens = data["open"].to_numpy(dtype=float)
        closes = data["close"].to_numpy(dtype=float)
        atr = indicators["atr"].to_numpy(dtype=float)
        votes = candidate.to_numpy(dtype=int)
        raw = np.zeros(len(data), dtype=int)
        position = desired = age = 0
        entry = stop = 0.0
        armed = False
        for i in range(len(data)):
            exited = False
            # A previous close's decision is filled at this bar's open.
            if position == 0 and desired:
                if i > 0 and np.isfinite(opens[i]) and opens[i] > 0 and np.isfinite(atr[i - 1]):
                    position = desired
                    entry = opens[i]
                    stop = entry - position * params.initial_stop_atr * atr[i - 1]
                    age = 0
                    armed = False
                else:
                    desired = 0
            desired = position
            if position:
                age += 1
                price = closes[i]
                # Stops are close-tested; an exit fills at the following open.
                if not np.isfinite(price) or position * (price - stop) <= 0 or age >= params.hold_bars:
                    position = desired = 0
                    exited = True
                else:
                    armed = armed or position * (price / entry - 1) >= params.breakeven_pct
                    if armed:
                        # Long stop only rises; short stop only falls.
                        stop = max(stop, entry) if position == 1 else min(stop, entry)
                        if np.isfinite(atr[i]) and atr[i] > 0:
                            trail = price - position * params.trail_atr * atr[i]
                            stop = max(stop, trail) if position == 1 else min(stop, trail)
            if position == 0 and not exited:
                desired = int(votes[i])
            raw[i] = desired
        df = pd.DataFrame({"signal": raw, "size": 1.0}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check():
    params = StreakParams()
    strategy = GeneratedStrategy()
    index = pd.date_range("2020-01-01", periods=7, freq="D")
    for direction in (1, -1):
        closes = np.array([100, 101, 99, 99, 99, 99, 99], dtype=float)
        if direction == -1:
            closes = 200 - closes
        data = pd.DataFrame({"open": 100.0, "high": closes + 2,
                             "low": closes - 2, "close": closes,
                             "volume": 1000.0}, index=index)
        ind = pd.DataFrame({"up_streak": params.streak_bars if direction == 1 else 0,
                            "down_streak": params.streak_bars if direction == -1 else 0,
                            "efficiency": 0.8, "pressure": direction * 0.5,
                            "atr": 1.0}, index=index)
        ind.loc[index[1:], "pressure"] = 0.0
        signal = strategy.generate_signals(data, ind, None, params).data["signal"]
        assert signal.tolist() == [0, direction, direction, 0, 0, 0, 0]
        data["close"] = 100.0
        signal = strategy.generate_signals(data, ind, None, params).data["signal"]
        assert signal.tolist() == [0, direction, direction, direction, direction, 0, 0]
        ind["pressure"] = -direction * 0.5
        assert strategy.generate_signals(data, ind, None, params).data["signal"].eq(0).all()
    derived = strategy.indicators(data, params)
    assert derived.index.equals(data.index)
    assert strategy.generate_signals(data, derived, None, params).data["signal"].eq(0).all()
    assert strategy.warmup_bars(params) == 21


if __name__ == "__main__":
    _self_check()
