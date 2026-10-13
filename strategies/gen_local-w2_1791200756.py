from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class RangeRecoveryParams:
    peak_window: int = 40
    pressure_window: int = 4
    atr_window: int = 14
    min_drawdown: float = 0.03
    recovery_share: float = 0.65
    lower_decay: float = 0.55
    activation_pct: float = 0.015
    initial_stop_atr: float = 2.5
    trail_atr: float = 2.5
    max_hold: int = 10

    def __post_init__(self):
        if min(self.peak_window, self.pressure_window, self.atr_window, self.max_hold) < 1:
            raise ValueError("Windows and holding period must be positive.")
        if not 0.0 < self.min_drawdown < 1.0 or not 0.0 < self.recovery_share < 1.0:
            raise ValueError("Drawdown and recovery thresholds must lie in (0, 1).")
        if not 0.0 < self.lower_decay < 1.0:
            raise ValueError("Lower decay must lie in (0, 1).")
        if min(self.activation_pct, self.initial_stop_atr, self.trail_atr) <= 0:
            raise ValueError("Stop parameters must be positive.")


class GeneratedStrategy(BaseStrategy[RangeRecoveryParams]):
    strategy_id = "gen_local-w2_1791200756"

    @classmethod
    def params_type(cls):
        return RangeRecoveryParams

    def warmup_bars(self, params):
        return int(max(params.peak_window, 2 * params.pressure_window + 1, params.atr_window + 1))

    def indicators(self, data, params):
        out = pd.DataFrame(index=data.index)
        out["previous_close"] = data["close"].shift(1)
        out["previous_high"] = data["high"].shift(1)
        out["previous_low"] = data["low"].shift(1)
        out["range"] = (data["high"] - data["low"]).clip(lower=0)
        out["true_range"] = pd.concat([
            out["range"],
            (data["high"] - out["previous_close"]).abs(),
            (data["low"] - out["previous_close"]).abs(),
        ], axis=1).max(axis=1)
        out["atr"] = out["true_range"].rolling(params.atr_window, min_periods=params.atr_window).mean()
        denominator = out["true_range"].replace(0, np.nan)
        out["upper_extension"] = (data["high"] - out["previous_high"]).clip(lower=0) / denominator
        out["lower_extension"] = (out["previous_low"] - data["low"]).clip(lower=0) / denominator
        out["upper_pressure"] = out["upper_extension"].rolling(params.pressure_window, min_periods=params.pressure_window).mean()
        out["lower_pressure"] = out["lower_extension"].rolling(params.pressure_window, min_periods=params.pressure_window).mean()
        out["prior_lower_pressure"] = out["lower_pressure"].shift(params.pressure_window)
        out["recovery_share"] = out["upper_pressure"] / (out["upper_pressure"] + out["lower_pressure"]).replace(0, np.nan)
        out["prior_recovery_share"] = out["recovery_share"].shift(1)
        out["lower_ratio"] = out["lower_pressure"] / out["prior_lower_pressure"].replace(0, np.nan)
        out["peak"] = data["close"].rolling(params.peak_window, min_periods=params.peak_window).max()
        out["drawdown"] = 1.0 - data["close"] / out["peak"].replace(0, np.nan)
        out["close_location"] = (data["close"] - data["low"]) / out["range"].replace(0, np.nan)
        out["strength"] = (
            0.5 * out["recovery_share"].clip(0, 1)
            + 0.5 * (1.0 - out["lower_ratio"].clip(0, 1))
        ).fillna(0).clip(0, 1)
        out["entry"] = (
            (out["drawdown"] >= params.min_drawdown)
            & (out["prior_lower_pressure"] >= 0.10)
            & (out["lower_ratio"] <= params.lower_decay)
            & (out["recovery_share"] >= params.recovery_share)
            & (out["prior_recovery_share"] < params.recovery_share)
            & (out["close_location"] >= 0.65)
            & (out["upper_extension"] > 0)
            & (out["atr"] > 0)
        ).fillna(False)
        return out.replace([np.inf, -np.inf], np.nan)

    def generate_signals(self, data, indicators, ctx, params):
        n = len(data)
        signals = np.zeros(n, dtype=int)
        sizes = np.full(n, 0.25, dtype=float)
        opens = data["open"].to_numpy(dtype=float)
        highs = data["high"].to_numpy(dtype=float)
        lows = data["low"].to_numpy(dtype=float)
        closes = data["close"].to_numpy(dtype=float)
        atr = indicators["atr"].to_numpy(dtype=float)
        entries = indicators["entry"].fillna(False).to_numpy(dtype=bool)
        strengths = indicators["strength"].fillna(0).clip(0, 1).to_numpy(dtype=float)
        active = False
        pending = False
        armed = False
        entry = stop = peak = 0.0
        position_size = 0.25
        held = 0
        for i in range(n):
            if pending:
                pending = False
                if np.isfinite(opens[i]) and opens[i] > 0 and np.isfinite(atr[i - 1]) and atr[i - 1] > 0:
                    active = True
                    armed = False
                    held = 0
                    entry = peak = opens[i]
                    stop = entry - params.initial_stop_atr * atr[i - 1]
                else:
                    continue
            if active:
                held += 1
                sizes[i] = position_size
                valid = np.isfinite([opens[i], highs[i], lows[i], closes[i], atr[i]]).all() and atr[i] > 0
                # Stops are close-time exit decisions, executed at the next open.
                # Test the existing stop before ratcheting with this bar's high.
                if not valid or lows[i] <= stop or held >= params.max_hold:
                    active = False
                    continue
                peak = max(peak, highs[i])
                if peak >= entry * (1.0 + params.activation_pct):
                    armed = True
                if armed:
                    stop = max(stop, entry, peak - params.trail_atr * atr[i])
                    if closes[i] <= stop:
                        active = False
                        continue
                signals[i] = 1
            elif entries[i] and np.isfinite(atr[i]) and atr[i] > 0:
                position_size = float(0.25 + 0.70 * strengths[i])
                signals[i] = 1
                sizes[i] = position_size
                pending = True
        df = pd.DataFrame({"signal": signals, "size": sizes}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = df["size"].shift(1).fillna(0.25).astype(float)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check():
    index = pd.date_range("2020-01-01", periods=5)
    data = pd.DataFrame({
        "open": [100.0] * 5,
        "high": [101.0, 104.0, 105.0, 104.0, 104.0],
        "low": [99.0, 99.0, 102.0, 101.0, 102.0],
        "close": [100.0, 103.0, 104.0, 103.0, 103.0],
        "volume": [1000.0] * 5,
    }, index=index)
    indicators = pd.DataFrame({
        "atr": [1.0, 1.0, 10.0, 10.0, 10.0],
        "entry": [True, False, False, False, False],
        "strength": [0.8] * 5,
    }, index=index)
    result = GeneratedStrategy().generate_signals(data, indicators, None, RangeRecoveryParams()).data
    assert result["signal"].tolist() == [0, 1, 1, 1, 0]
    assert np.isclose(result["size"].iloc[1], 0.81)
    assert (result["size"] > 0).all()
    flat = data.copy()
    flat[["open", "high", "low", "close"]] = 100.0
    strategy = GeneratedStrategy()
    params = RangeRecoveryParams()
    result = strategy.generate_signals(flat, strategy.indicators(flat, params), None, params).data
    assert not result["signal"].any()


if __name__ == "__main__":
    _self_check()
