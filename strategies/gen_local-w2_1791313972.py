from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class VolatilityEchoParams:
    vol_window: int = 5
    corr_window: int = 20
    baseline_window: int = 40
    atr_window: int = 14
    corr_min: float = 0.25
    expansion_ratio: float = 1.15
    contraction_scale: float = 0.08
    breakeven_pct: float = 0.012
    trail_atr: float = 2.2
    initial_stop_atr: float = 2.5
    max_hold: int = 10
    min_size: float = 0.20
    max_size: float = 0.95

    def __post_init__(self):
        if min(self.vol_window, self.corr_window, self.baseline_window, self.atr_window) < 2:
            raise ValueError("Indicator windows must be at least two.")
        if not (0 < self.corr_min < 1 and self.expansion_ratio > 0):
            raise ValueError("Invalid entry thresholds.")
        if not (self.contraction_scale > 0 and self.breakeven_pct > 0):
            raise ValueError("Contraction scale and breakeven threshold must be positive.")
        if not (self.trail_atr > 0 and self.initial_stop_atr > 0 and self.max_hold >= 1):
            raise ValueError("Invalid exit parameters.")
        if not (0 < self.min_size <= self.max_size <= 1):
            raise ValueError("Sizes must satisfy 0 < min_size <= max_size <= 1.")


class GeneratedStrategy(BaseStrategy[VolatilityEchoParams]):
    strategy_id = "gen_local-w2_1791313972"

    @classmethod
    def params_type(cls):
        return VolatilityEchoParams

    def warmup_bars(self, params):
        return int(max(params.vol_window + params.corr_window + 1,
                       params.baseline_window + 1, params.atr_window + 1))

    def indicators(self, data, params):
        close = data["close"].where(data["close"] > 0)
        returns = close.pct_change(fill_method=None).replace([np.inf, -np.inf], np.nan)
        vol = returns.rolling(params.vol_window).std(ddof=0)
        baseline = returns.rolling(params.baseline_window).std(ddof=0)
        lag_vol = vol.shift(1)
        rho = vol.rolling(params.corr_window).corr(lag_vol).clip(-1, 1)
        forecast = (vol + rho * (vol - lag_vol)).clip(lower=0)
        expansion = vol.div(baseline.where(baseline > 0))
        decay = (vol - forecast).div(vol.where(vol > 0))
        previous_close = close.shift(1)
        true_range = pd.concat([
            data["high"] - data["low"],
            (data["high"] - previous_close).abs(),
            (data["low"] - previous_close).abs()
        ], axis=1).max(axis=1)
        atr = true_range.rolling(params.atr_window).mean()
        out = pd.DataFrame({
            "returns": returns, "vol": vol, "baseline": baseline,
            "lag_vol": lag_vol, "rho": rho, "forecast": forecast,
            "expansion": expansion, "decay": decay,
            "true_range": true_range, "atr": atr
        }, index=data.index).replace([np.inf, -np.inf], np.nan)
        out["ready"] = out.notna().all(axis=1) & (atr > 0)
        out["candidate"] = (
            out["ready"] & (rho > params.corr_min)
            & (expansion > params.expansion_ratio)
            & (decay > 0) & (returns > 0)
        ).fillna(False)
        out["conviction"] = (rho.clip(lower=0) * decay.clip(lower=0)
                             * expansion / params.contraction_scale).clip(0, 1).fillna(0)
        return out

    def generate_signals(self, data, indicators, ctx: StrategyContext, params):
        n = len(data)
        signals = np.zeros(n, dtype=int)
        sizes = np.full(n, params.min_size, dtype=float)
        active = False
        armed = False
        entry = np.nan
        stop = np.nan
        peak = np.nan
        held = 0
        position_size = params.min_size
        opens = data["open"].to_numpy(dtype=float)
        closes = data["close"].to_numpy(dtype=float)
        atrs = indicators["atr"].to_numpy(dtype=float)
        candidates = indicators["candidate"].fillna(False).to_numpy(dtype=bool)
        readiness = indicators["ready"].fillna(False).to_numpy(dtype=bool)
        conviction = indicators["conviction"].fillna(0).to_numpy(dtype=float)

        for i in range(n):
            if active:
                # A decision at i-1 enters at the observed open of i.
                if np.isnan(entry):
                    entry = opens[i]
                    peak = entry
                    stop = entry - params.initial_stop_atr * atrs[i - 1]
                    held = 0
                held += 1
                valid = (readiness[i] and np.isfinite(entry) and entry > 0
                         and np.isfinite(closes[i]) and closes[i] > 0)
                if not valid:
                    active = False
                else:
                    # Stops are evaluated at closes; exits fill next bar.
                    breached = closes[i] <= stop
                    peak = max(peak, closes[i])
                    if closes[i] >= entry * (1 + params.breakeven_pct):
                        armed = True
                    if armed:
                        stop = max(stop, entry, peak - params.trail_atr * atrs[i])
                    if breached or closes[i] <= stop or held >= params.max_hold:
                        active = False
                if not active:
                    entry = np.nan
                    armed = False
            elif readiness[i] and candidates[i]:
                active = True
                entry = np.nan
                armed = False
                position_size = params.min_size + (
                    params.max_size - params.min_size
                ) * float(np.clip(conviction[i], 0, 1))
            if active:
                signals[i] = 1
                sizes[i] = position_size

        df = pd.DataFrame({"signal": signals, "size": sizes}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = df["size"].shift(1).fillna(params.min_size).astype(float)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check():
    params = VolatilityEchoParams()
    strategy = GeneratedStrategy()
    index = pd.date_range("2020-01-01", periods=7)
    data = pd.DataFrame({
        "open": [100, 100, 101, 103, 102, 100.5, 100],
        "high": [101, 102, 104, 104, 103, 102, 101],
        "low": [99, 99, 100, 101, 100, 99, 99],
        "close": [100, 101, 103, 102, 100.5, 100, 100],
        "volume": 1000000
    }, index=index)
    derived = pd.DataFrame({
        "atr": [1, 1, 1, 5, 5, 5, 5],
        "ready": True,
        "candidate": [True, False, False, False, False, False, False],
        "conviction": 0.7
    }, index=index)
    result = strategy.generate_signals(data, derived, None, params).data
    assert result["signal"].tolist() == [0, 1, 1, 1, 1, 0, 0]
    assert np.isclose(result["size"].iloc[1], 0.725)
    assert (result["size"] > 0).all()
    constant = pd.DataFrame({
        "open": 100.0, "high": 100.0, "low": 100.0,
        "close": 100.0, "volume": 1000000
    }, index=pd.date_range("2020-01-01", periods=80))
    derived = strategy.indicators(constant, params)
    assert not derived["candidate"].any()
    assert not strategy.generate_signals(constant, derived, None, params).data["signal"].any()


if __name__ == "__main__":
    _self_check()
