from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class ParticipationParams:
    regime_window: int = 12
    atr_window: int = 14
    concentration_threshold: float = 1.35
    volume_threshold: float = 1.3
    move_threshold: float = 0.6
    breakeven_pct: float = 0.01
    trail_atr: float = 1.5
    holding_days: int = 4


class GeneratedStrategy(BaseStrategy[ParticipationParams]):
    strategy_id = "gen_local-w2_1791236937"

    @classmethod
    def params_type(cls):
        return ParticipationParams

    @staticmethod
    def warmup_bars(params):
        return int(max(params.regime_window, params.atr_window) + 1)

    @staticmethod
    def indicators(data, params):
        if (params.regime_window < 2 or params.atr_window < 2
                or not 3 <= params.holding_days <= 5
                or params.concentration_threshold <= 1
                or params.volume_threshold <= 0
                or params.move_threshold <= 0
                or params.breakeven_pct <= 0 or params.trail_atr <= 0):
            raise ValueError("Invalid strategy parameters")
        out = pd.DataFrame(index=data.index)
        close = data["close"]
        previous = close.shift(1)
        volume = data["volume"].where(data["volume"] > 0)
        true_range = pd.concat([
            data["high"] - data["low"],
            (data["high"] - previous).abs(),
            (data["low"] - previous).abs(),
        ], axis=1).max(axis=1)
        out["atr"] = true_range.rolling(params.atr_window).mean()
        prior_volume = volume.shift(1)
        total = prior_volume.rolling(params.regime_window).sum()
        squares = prior_volume.pow(2).rolling(params.regime_window).sum()
        out["concentration"] = params.regime_window * squares / total.pow(2)
        out["relative_volume"] = volume / prior_volume.rolling(params.regime_window).mean()
        out["move"] = (close - previous) / out["atr"].shift(1).replace(0, np.nan)
        span = (data["high"] - data["low"]).replace(0, np.nan)
        out["close_location"] = (close - data["low"]) / span
        out["body"] = (close - data["open"]) / span
        out = out.replace([np.inf, -np.inf], np.nan)
        valid = out.notna().all(axis=1)
        confirmed = valid & (out["relative_volume"] >= params.volume_threshold) & (
            out["move"].abs() >= params.move_threshold)
        distributed = out["concentration"] < params.concentration_threshold
        continuation = distributed & (out["move"] * out["body"] > 0)
        rejection = ~distributed & (
            ((out["move"] > 0) & (out["close_location"] <= 0.4)) |
            ((out["move"] < 0) & (out["close_location"] >= 0.6)))
        direction = np.sign(out["move"]).fillna(0)
        out["candidate"] = np.select(
            [confirmed & continuation, confirmed & rejection],
            [direction, -direction], default=0).astype(int)
        evidence = (
            (out["relative_volume"] / params.volume_threshold - 1).clip(0, 1) +
            (out["move"].abs() / params.move_threshold - 1).clip(0, 1)
        ) / 2
        out["conviction_size"] = (0.25 + 0.7 * evidence).fillna(0.25).clip(0.25, 0.95)
        return out

    @staticmethod
    def generate_signals(data, indicators, ctx: StrategyContext, params):
        raw = np.zeros(len(data), dtype=int)
        sizes = np.full(len(data), 0.25, dtype=float)
        direction = 0
        pending = False
        armed = False
        entry = stop = exposure = 0.0
        age = 0
        for i in range(len(data)):
            price = float(data["close"].iloc[i])
            atr = float(indicators["atr"].iloc[i])
            exited = False
            if pending:
                entry = float(data["open"].iloc[i])
                prior_atr = float(indicators["atr"].iloc[i - 1])
                pending = False
                armed = False
                age = 0
                if not np.isfinite(entry) or entry <= 0 or not np.isfinite(prior_atr) or prior_atr <= 0:
                    direction = 0
                    exited = True
                else:
                    stop = entry - direction * params.trail_atr * prior_atr
            if direction:
                age += 1
                extreme = float(data["low" if direction == 1 else "high"].iloc[i])
                if (not np.isfinite(price) or not np.isfinite(extreme)
                        or not np.isfinite(atr) or atr <= 0
                        or direction * (extreme - stop) <= 0
                        or age >= params.holding_days):
                    direction = 0
                    exited = True
                else:
                    if direction * (price / entry - 1) >= params.breakeven_pct:
                        armed = True
                    if armed:
                        proposed = price - direction * params.trail_atr * atr
                        # Ratchet up for longs; mirror downward for shorts.
                        stop = max(stop, entry, proposed) if direction == 1 else min(stop, entry, proposed)
            if direction == 0 and not exited:
                candidate = int(indicators["candidate"].iloc[i])
                if candidate and np.isfinite(price) and price > 0 and np.isfinite(atr) and atr > 0:
                    direction = candidate
                    exposure = float(indicators["conviction_size"].iloc[i])
                    pending = True
            raw[i] = direction
            if direction:
                sizes[i] = exposure
        df = pd.DataFrame({"signal": raw, "size": sizes}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = df["size"].shift(1).fillna(0.25).astype(float)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check():
    index = pd.date_range("2020-01-01", periods=10)
    data = pd.DataFrame({
        "open": [100.0] * 10,
        "high": [100, 101.4, 101.8, 101.8, 100, 100, 100.2, 99.5, 99.6, 100],
        "low": [100, 99.5, 100.7, 100.4, 100, 100, 98.6, 98.2, 98.2, 100],
        "close": [100, 101.2, 101.6, 101, 100, 100, 98.8, 98.4, 99, 100],
        "volume": [1000.0] * 10,
    }, index=index)
    features = pd.DataFrame({
        "atr": 1.0,
        "candidate": [1, 0, 0, 0, 0, -1, 0, 0, 0, 0],
        "conviction_size": [0.7] * 5 + [0.6] * 5,
    }, index=index)
    params = ParticipationParams(trail_atr=1.0)
    result = GeneratedStrategy.generate_signals(data, features, None, params).data
    assert result["signal"].tolist() == [0, 1, 1, 1, 0, 0, -1, -1, -1, 0]
    assert np.allclose(result["size"].iloc[1:4], 0.7)
    assert np.allclose(result["size"].iloc[6:9], 0.6)
    flat = pd.DataFrame({"open": 100.0, "high": 100.1, "low": 99.9,
                         "close": 100.0, "volume": 1000.0}, index=index)
    features["candidate"] = [1] + [0] * 9
    result = GeneratedStrategy.generate_signals(flat, features, None, params).data
    assert result["signal"].tolist() == [0, 1, 1, 1, 1, 0, 0, 0, 0, 0]
    assert GeneratedStrategy.indicators(flat, params)["candidate"].eq(0).all()


if __name__ == "__main__":
    _self_check()
