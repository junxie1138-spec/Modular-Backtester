from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class Params:
    fast_roc: int = 3
    slow_roc: int = 10
    acceleration_lag: int = 2
    history: int = 126
    handoff_window: int = 5
    percentile: float = 0.80
    atr_window: int = 14
    breakeven_pct: float = 0.015
    trail_atr: float = 2.5
    max_hold: int = 20

    def __post_init__(self):
        windows = (self.fast_roc, self.slow_roc, self.acceleration_lag,
                   self.history, self.handoff_window, self.atr_window, self.max_hold)
        if any(w <= 0 for w in windows):
            raise ValueError("Lookbacks and holding period must be positive.")
        if self.fast_roc >= self.slow_roc or not 0 < self.percentile < 1:
            raise ValueError("Require fast_roc < slow_roc and 0 < percentile < 1.")
        if not np.isfinite(self.breakeven_pct) or self.breakeven_pct <= 0:
            raise ValueError("breakeven_pct must be finite and positive.")
        if not np.isfinite(self.trail_atr) or self.trail_atr <= 0:
            raise ValueError("trail_atr must be finite and positive.")


class GeneratedStrategy(BaseStrategy[Params]):
    strategy_id = "gen_local_1791276560"

    @classmethod
    def params_type(cls):
        return Params

    def warmup_bars(self, params: Params) -> int:
        return int(max(
            params.fast_roc + params.acceleration_lag + params.history
            + params.handoff_window + 1,
            params.slow_roc + params.acceleration_lag + params.history + 1,
            params.atr_window + 1,
        ))

    def indicators(self, data: pd.DataFrame, params: Params) -> pd.DataFrame:
        out = pd.DataFrame(index=data.index)
        close = data["close"].where(data["close"] > 0)
        out["fast_roc"] = close.pct_change(params.fast_roc, fill_method=None)
        out["slow_roc"] = close.pct_change(params.slow_roc, fill_method=None)
        out["fast_acceleration"] = out["fast_roc"].diff(params.acceleration_lag)
        out["slow_acceleration"] = out["slow_roc"].diff(params.acceleration_lag)
        for speed in ("fast", "slow"):
            out[speed + "_threshold"] = (
                out[speed + "_acceleration"].shift(1)
                .rolling(params.history, min_periods=params.history)
                .quantile(params.percentile)
            )
        out["fast_shock"] = (
            (out["fast_acceleration"] > out["fast_threshold"])
            & (out["fast_acceleration"] > 0)
        )
        out["recent_shock"] = (
            out["fast_shock"].astype(float).shift(1)
            .rolling(params.handoff_window, min_periods=params.handoff_window).max()
        )
        previous_close = close.shift(1)
        out["true_range"] = pd.concat([
            data["high"] - data["low"],
            (data["high"] - previous_close).abs(),
            (data["low"] - previous_close).abs(),
        ], axis=1).max(axis=1)
        out["atr"] = out["true_range"].rolling(
            params.atr_window, min_periods=params.atr_window
        ).mean()
        out["ready"] = out[[
            "fast_roc", "slow_roc", "fast_acceleration", "slow_acceleration",
            "fast_threshold", "slow_threshold", "recent_shock", "atr",
        ]].replace([np.inf, -np.inf], np.nan).notna().all(axis=1)
        out["entry"] = (
            out["ready"] & (out["atr"] > 0) & (out["recent_shock"] == 1)
            & (out["slow_acceleration"] > out["slow_threshold"])
            & (out["slow_acceleration"] > 0)
            & (out["fast_acceleration"] <= out["fast_threshold"])
            & (out["fast_roc"] > 0) & (out["slow_roc"] > 0)
        ).fillna(False)
        return out

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame,
                         ctx: StrategyContext, params: Params) -> SignalFrame:
        signals = np.zeros(len(data), dtype=int)
        opens = data["open"].to_numpy(dtype=float)
        closes = data["close"].to_numpy(dtype=float)
        atrs = indicators["atr"].to_numpy(dtype=float)
        entries = indicators["entry"].fillna(False).to_numpy(dtype=bool)
        holding = pending = armed = False
        entry_price = stop = 0.0
        age = 0
        for i in range(len(data)):
            if pending:
                pending = False
                # Entry reference is this bar's open, after the prior close decision.
                if np.isfinite(opens[i]) and opens[i] > 0:
                    entry_price = opens[i]
                    stop = entry_price - params.trail_atr * atrs[i - 1]
                    holding, armed, age = True, False, 0
            if holding:
                age += 1
                close = closes[i]
                # Stops are evaluated at closes; exits execute at the next open.
                stopped = np.isfinite(close) and close <= stop
                if stopped or age >= params.max_hold:
                    holding = False
                elif np.isfinite(close) and close > 0:
                    if close >= entry_price * (1.0 + params.breakeven_pct):
                        armed = True
                    if armed:
                        stop = max(stop, entry_price)
                        if np.isfinite(atrs[i]) and atrs[i] > 0:
                            stop = max(stop, close - params.trail_atr * atrs[i])
                signals[i] = int(holding)
            elif entries[i] and np.isfinite(closes[i]) and closes[i] > 0:
                pending = True
                signals[i] = 1
        df = pd.DataFrame({"signal": signals, "size": 0.95}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


if __name__ == "__main__":
    index = pd.date_range("2020-01-01", periods=6)
    data = pd.DataFrame({
        "open": 100.0, "high": [101, 101, 104, 105, 102, 101],
        "low": [99, 99, 102, 103, 100, 99],
        "close": [100, 100, 103, 104, 101, 100], "volume": 1000.0,
    }, index=index)
    indicators = pd.DataFrame({
        "atr": 1.0, "entry": [True, False, False, False, False, False],
    }, index=index)
    strategy = GeneratedStrategy()
    result = strategy.generate_signals(data, indicators, None, Params()).data
    assert result["signal"].tolist() == [0, 1, 1, 1, 1, 0]
    prefix = strategy.generate_signals(
        data.iloc[:4], indicators.iloc[:4], None, Params()
    ).data
    assert prefix["signal"].tolist() == result["signal"].iloc[:4].tolist()
