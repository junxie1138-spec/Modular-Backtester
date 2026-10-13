from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class ObserverParams:
    level_gain: float = 0.35
    velocity_gain: float = 0.06
    noise_window: int = 20
    atr_window: int = 14
    breakout_z: float = 1.5
    stop_atr: float = 1.5
    hold_days: int = 2
    minimum_size: float = 0.25


class GeneratedStrategy(BaseStrategy[ObserverParams]):
    strategy_id = "gen_local_1791205869"

    @classmethod
    def params_type(cls):
        return ObserverParams

    @staticmethod
    def warmup_bars(params: ObserverParams) -> int:
        gain = min(max(params.level_gain, 0.001), max(params.velocity_gain, 0.001))
        return int(max(params.noise_window + 1, params.atr_window + 1, np.ceil(5.0 / gain)))

    @staticmethod
    def indicators(data: pd.DataFrame, params: ObserverParams) -> pd.DataFrame:
        out = pd.DataFrame(index=data.index)
        close = data["close"].astype(float)
        previous_close = close.shift(1)
        true_range = pd.concat([
            data["high"] - data["low"],
            (data["high"] - previous_close).abs(),
            (data["low"] - previous_close).abs(),
        ], axis=1).max(axis=1)
        out["atr"] = true_range.rolling(params.atr_window, min_periods=params.atr_window).mean()
        prices = close.to_numpy()
        forecasts = np.full(len(data), np.nan)
        errors = np.full(len(data), np.nan)
        velocities = np.full(len(data), np.nan)
        level = np.nan
        velocity = 0.0
        alpha = float(np.clip(params.level_gain, 0.001, 1.0))
        beta = float(np.clip(params.velocity_gain, 0.001, 1.0))
        for i, price in enumerate(prices):
            if not np.isfinite(price):
                continue
            if not np.isfinite(level):
                level = price
                velocities[i] = velocity
                continue
            prediction = level + velocity
            error = price - prediction
            level = prediction + alpha * error
            velocity += beta * error
            forecasts[i] = prediction
            errors[i] = error
            velocities[i] = velocity
        out["forecast"] = forecasts
        out["forecast_error"] = errors
        out["velocity"] = velocities
        out["previous_velocity"] = out["velocity"].shift(1)
        out["forecast_noise"] = out["forecast_error"].rolling(
            params.noise_window, min_periods=params.noise_window
        ).std(ddof=0).shift(1)
        denominator = out["forecast_noise"].where(out["forecast_noise"] > 0)
        out["surprise_z"] = (out["forecast_error"] / denominator).replace([np.inf, -np.inf], np.nan)
        out["entry"] = (
            (out["velocity"] > 0)
            & (out["previous_velocity"] <= 0)
            & (out["surprise_z"] > params.breakout_z)
            & (out["atr"] > 0)
        ).fillna(False)
        return out

    @staticmethod
    def generate_signals(data: pd.DataFrame, indicators: pd.DataFrame,
                         ctx: StrategyContext, params: ObserverParams) -> SignalFrame:
        df = pd.DataFrame(index=data.index)
        signals = np.zeros(len(data), dtype=int)
        sizes = np.ones(len(data), dtype=float)
        opens = data["open"].to_numpy(dtype=float)
        closes = data["close"].to_numpy(dtype=float)
        atrs = indicators["atr"].to_numpy(dtype=float)
        entries = indicators["entry"].to_numpy(dtype=bool)
        strengths = indicators["surprise_z"].to_numpy(dtype=float)
        active = False
        pending_entry = False
        entry_bar = -1
        entry_atr = np.nan
        stop_price = np.nan
        position_size = 1.0
        hold_days = int(np.clip(params.hold_days, 1, 2))
        minimum_size = float(np.clip(params.minimum_size, 0.01, 1.0))
        warmup = GeneratedStrategy.warmup_bars(params)
        for i in range(len(data)):
            if active:
                if pending_entry:
                    if np.isfinite(opens[i]) and opens[i] > 0:
                        entry_bar = i
                        stop_price = opens[i] - params.stop_atr * entry_atr
                        pending_entry = False
                    else:
                        active = False
                if active and not pending_entry:
                    stop_hit = np.isfinite(closes[i]) and closes[i] < stop_price
                    expired = i - entry_bar + 1 >= hold_days
                    if stop_hit or expired:
                        active = False
                if active:
                    signals[i] = 1
                    sizes[i] = position_size
                continue
            if i >= warmup and entries[i] and np.isfinite(strengths[i]):
                active = True
                pending_entry = True
                entry_atr = atrs[i]
                excess = strengths[i] - params.breakout_z
                position_size = float(np.clip(
                    minimum_size + (1.0 - minimum_size) * excess / max(params.breakout_z, 0.01),
                    minimum_size, 1.0,
                ))
                signals[i] = 1
                sizes[i] = position_size
        df["signal"] = signals
        df["size"] = sizes
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = df["size"].shift(1).fillna(1.0).astype(float)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check() -> None:
    params = ObserverParams()
    n = GeneratedStrategy.warmup_bars(params) + 8
    index = pd.date_range("2020-01-01", periods=n, freq="B")
    data = pd.DataFrame({"open": 100.0, "high": 101.0, "low": 99.0,
                         "close": 100.0, "volume": 1000.0}, index=index)
    derived = GeneratedStrategy.indicators(data, params)
    derived["entry"] = False
    derived["surprise_z"] = 3.0
    derived["atr"] = 2.0
    decision = GeneratedStrategy.warmup_bars(params)
    derived.iloc[decision, derived.columns.get_loc("entry")] = True
    result = GeneratedStrategy.generate_signals(data, derived, None, params).data
    assert result["signal"].iloc[decision:decision + 4].tolist() == [0, 1, 1, 0]
    assert result["signal"].isin([0, 1]).all()
    assert (result["size"] > 0).all()
    data.iloc[decision + 1, data.columns.get_loc("close")] = 96.0
    stopped = GeneratedStrategy.generate_signals(data, derived, None, params).data
    assert stopped["signal"].iloc[decision:decision + 3].tolist() == [0, 1, 0]
    derived["surprise_z"] = 2.0
    smaller = GeneratedStrategy.generate_signals(data, derived, None, params).data
    assert smaller["size"].iloc[decision + 1] < stopped["size"].iloc[decision + 1]


if __name__ == "__main__":
    _self_check()
