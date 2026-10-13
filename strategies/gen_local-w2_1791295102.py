from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class GapCancellationParams:
    lookback: int = 30
    gap_z: float = 1.5
    correlation_cutoff: float = -0.2
    target_vol: float = 0.12
    atr_bars: int = 14
    breakeven_pct: float = 0.015
    trail_atr: float = 2.0
    initial_stop_atr: float = 2.5
    max_hold: int = 10


class GeneratedStrategy(BaseStrategy[GapCancellationParams]):
    strategy_id = "gen_local-w2_1791295102"

    @classmethod
    def params_type(cls):
        return GapCancellationParams

    def warmup_bars(self, params):
        return int(max(params.lookback + 3, params.atr_bars + 1))

    def indicators(self, data, params):
        previous = data["close"].shift(1).where(lambda s: s > 0)
        opening = data["open"].where(lambda s: s > 0)
        closing = data["close"].where(lambda s: s > 0)
        gap = np.log(opening / previous)
        session = np.log(closing / opening)
        gv = gap.rolling(params.lookback, min_periods=params.lookback).var()
        sv = session.rolling(params.lookback, min_periods=params.lookback).var()
        covariance = gap.rolling(params.lookback, min_periods=params.lookback).cov(session)
        correlation = (covariance / np.sqrt(gv * sv).replace(0, np.nan)).shift(1)
        gap_scale = np.sqrt(gv).shift(1).replace(0, np.nan)
        # Do not mistake overnight/session cancellation for low position risk.
        risk_vol = np.sqrt((gv + sv + 2 * covariance.abs()).clip(lower=0) * 252)
        true_range = pd.concat([
            data["high"] - data["low"],
            (data["high"] - previous).abs(),
            (data["low"] - previous).abs(),
        ], axis=1).max(axis=1)
        atr = true_range.rolling(params.atr_bars, min_periods=params.atr_bars).mean()
        midpoint = (opening + previous) / 2
        shock = (gap.abs() >= params.gap_z * gap_scale) & (correlation <= params.correlation_cutoff)
        fade = -np.sign(gap)
        first = shock & (fade * (closing - midpoint) > 0)
        # Shock-day close and following close must both retain the retracement.
        second = first.shift(1).fillna(False).astype(bool) & (fade.shift(1) * (closing - midpoint.shift(1)) > 0)
        direction = fade.shift(1).where(second, 0).fillna(0).astype(int)
        size = (params.target_vol / risk_vol.replace(0, np.nan)).clip(0.05, 0.95)
        return pd.DataFrame({
            "gap": gap,
            "session": session,
            "gap_variance": gv,
            "session_variance": sv,
            "covariance": covariance,
            "gap_scale": gap_scale,
            "correlation": correlation,
            "risk_vol": risk_vol,
            "true_range": true_range,
            "atr": atr,
            "midpoint": midpoint,
            "shock": shock,
            "fade_direction": fade,
            "first_confirmation": first,
            "second_confirmation": second,
            "direction": direction,
            "target_size": size,
        }, index=data.index).replace([np.inf, -np.inf], np.nan)

    def generate_signals(self, data, indicators, ctx, params):
        count = len(data)
        signals = np.zeros(count, dtype=int)
        sizes = np.full(count, 0.05, dtype=float)
        position = pending = age = 0
        entry = best = stop = 0.0
        held_size = 0.05
        armed = False
        for i in range(count):
            atr = float(indicators["atr"].iloc[i])
            close = float(data["close"].iloc[i])
            if pending:
                opening = float(data["open"].iloc[i])
                prior_atr = float(indicators["atr"].iloc[i - 1])
                if np.isfinite(opening) and opening > 0 and np.isfinite(prior_atr) and prior_atr > 0:
                    position = pending
                    entry = opening
                    best = position * entry
                    stop = best - params.initial_stop_atr * prior_atr
                    age = 0
                    armed = False
                pending = 0
            exited = False
            if position:
                age += 1
                valid = np.isfinite(close) and close > 0 and np.isfinite(atr) and atr > 0
                if valid:
                    signed_close = position * close
                    best = max(best, signed_close)
                    armed = armed or (best - position * entry >= entry * params.breakeven_pct)
                    if armed:
                        # Signed-price stops increase for both long and short positions.
                        # Stops are evaluated at closes; exits execute next bar.
                        stop = max(stop, position * entry, best - params.trail_atr * atr)
                    exited = signed_close <= stop
                if not valid or exited or age >= params.max_hold:
                    position = 0
                    exited = True
            if not position and not exited:
                direction = indicators["direction"].iloc[i]
                target_size = indicators["target_size"].iloc[i]
                if np.isfinite(direction) and direction in (-1, 1) and np.isfinite(target_size) and target_size > 0 and np.isfinite(atr) and atr > 0:
                    pending = int(direction)
                    held_size = float(target_size)
            signals[i] = position if position else pending
            sizes[i] = held_size
        df = pd.DataFrame({"signal": signals, "size": sizes}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = df["size"].shift(1).fillna(0.05).astype(float)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


if __name__ == "__main__":
    strategy = GeneratedStrategy()
    params = GapCancellationParams()
    index = pd.date_range("2020-01-01", periods=6)
    ctx = StrategyContext(symbol="SPY", timeframe="1d", warmup_bars=0)
    for side in (1, -1):
        closes = 100 + side * np.array([0, 0, 2, 4, 0, 0], dtype=float)
        data = pd.DataFrame({"open": 100.0, "high": closes + 1, "low": closes - 1, "close": closes, "volume": 1000.0}, index=index)
        derived = pd.DataFrame({"atr": 1.0, "direction": [0, side, 0, 0, 0, 0], "target_size": 0.5}, index=index)
        result = strategy.generate_signals(data, derived, ctx, params).data
        assert result["signal"].tolist() == [0, 0, side, side, side, 0]
        assert (result["size"] > 0).all()
    assert strategy.warmup_bars(params) >= params.lookback + 3
