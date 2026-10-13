from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class GeneratedParams:
    roc_length: int = 5
    acceleration_lag: int = 3
    volatility_window: int = 20
    percentile_window: int = 126
    entry_quantile: float = 0.85
    atr_window: int = 14
    trail_atr: float = 2.5
    target_volatility: float = 0.12
    max_hold: int = 10
    min_size: float = 0.05
    max_size: float = 0.95

    def __post_init__(self):
        windows = (self.roc_length, self.acceleration_lag,
                   self.volatility_window, self.percentile_window,
                   self.atr_window, self.max_hold)
        if any(not isinstance(x, int) or isinstance(x, bool) or x < 1 for x in windows):
            raise ValueError("Lookbacks and max_hold must be positive integers.")
        if self.volatility_window < 2 or self.percentile_window < 2:
            raise ValueError("Volatility and percentile windows must be at least two.")
        if not 0.5 < self.entry_quantile < 1.0:
            raise ValueError("entry_quantile must be between 0.5 and 1.")
        if not all(np.isfinite(x) and x > 0 for x in
                   (self.trail_atr, self.target_volatility, self.min_size, self.max_size)):
            raise ValueError("Risk and sizing parameters must be finite and positive.")
        if not self.min_size <= self.max_size <= 1.0:
            raise ValueError("Require 0 < min_size <= max_size <= 1.")


class GeneratedStrategy(BaseStrategy[GeneratedParams]):
    strategy_id = "gen_local_1791308411"

    @classmethod
    def params_type(cls):
        return GeneratedParams

    def warmup_bars(self, params: GeneratedParams) -> int:
        return int(max(
            max(params.roc_length + 1, params.volatility_window + 1)
            + params.acceleration_lag + params.percentile_window + 1,
            params.atr_window + 1,
        ))

    def indicators(self, data: pd.DataFrame, params: GeneratedParams) -> pd.DataFrame:
        out = pd.DataFrame(index=data.index)
        close = data["close"]
        out["return"] = close.pct_change(fill_method=None)
        out["roc"] = close.pct_change(params.roc_length, fill_method=None)
        out["acceleration"] = out["roc"].diff(params.acceleration_lag)
        out["volatility"] = out["return"].rolling(params.volatility_window).std(ddof=0)
        denominator = out["volatility"].where(out["volatility"] > 0)
        out["risk_adjusted_roc"] = out["roc"] / (denominator * np.sqrt(params.roc_length))
        out["risk_acceleration"] = out["risk_adjusted_roc"].diff(params.acceleration_lag)
        out["threshold"] = out["risk_acceleration"].shift(1).rolling(
            params.percentile_window
        ).quantile(params.entry_quantile)
        out["volatility_change"] = out["volatility"].diff(params.acceleration_lag)
        previous = close.shift(1)
        out["true_range"] = pd.concat(
            [data["high"] - data["low"],
             (data["high"] - previous).abs(),
             (data["low"] - previous).abs()], axis=1
        ).max(axis=1, skipna=False)
        out["atr"] = out["true_range"].rolling(params.atr_window).mean()
        out["risk_size"] = (
            params.target_volatility / (denominator * np.sqrt(252.0))
        ).clip(params.min_size, params.max_size)
        out = out.replace([np.inf, -np.inf], np.nan)
        out["entry"] = (
            (out["roc"] > 0)
            & (out["acceleration"] > 0)
            & (out["risk_acceleration"] > 0)
            & (out["risk_acceleration"] > out["threshold"])
            & (out["volatility_change"] < 0)
        ).fillna(False)
        return out

    def generate_signals(
        self, data: pd.DataFrame, indicators: pd.DataFrame,
        ctx: StrategyContext, params: GeneratedParams,
    ) -> SignalFrame:
        closes = data["close"].to_numpy(dtype=float)
        atr = indicators["atr"].to_numpy(dtype=float)
        entries = indicators["entry"].fillna(False).to_numpy(dtype=bool)
        sizes = indicators["risk_size"].to_numpy(dtype=float)
        raw = np.zeros(len(data), dtype=int)
        active = False
        entry_bar = -1
        highest = -np.inf
        stop = -np.inf
        for i in range(len(data)):
            valid = (np.isfinite(closes[i]) and closes[i] > 0
                     and np.isfinite(atr[i]) and atr[i] > 0
                     and np.isfinite(sizes[i]) and sizes[i] > 0)
            if active:
                if valid:
                    # First update is the first filled bar's close.
                    highest = max(highest, closes[i])
                    stop = max(stop, highest - params.trail_atr * atr[i])
                if not valid or closes[i] <= stop or i - entry_bar >= params.max_hold:
                    active = False
                else:
                    raw[i] = 1
            elif valid and entries[i]:
                active = True
                entry_bar = i
                highest = -np.inf
                stop = -np.inf
                raw[i] = 1
        df = pd.DataFrame(index=data.index)
        df["signal"] = raw
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = indicators["risk_size"].shift(1).fillna(params.min_size).clip(
            params.min_size, params.max_size
        ).astype(float)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check():
    params = GeneratedParams()
    index = pd.date_range("2020-01-01", periods=7)
    data = pd.DataFrame({"close": [100., 101., 104., 110., 101., 100., 100.]}, index=index)
    derived = pd.DataFrame({
        "atr": [1., 1., 1., 10., 10., 10., 10.],
        "entry": [True, False, False, False, False, False, False],
        "risk_size": [0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80],
    }, index=index)
    ctx = StrategyContext(symbol="SPY", timeframe="1d", warmup_bars=0)
    result = GeneratedStrategy().generate_signals(data, derived, ctx, params).data
    # Rising ATR must not lower the stop; signals and sizes are delayed once.
    assert result["signal"].tolist() == [0, 1, 1, 1, 1, 0, 0]
    assert np.isclose(result["size"].iloc[1], 0.20)
    assert (result["size"] > 0).all()


if __name__ == "__main__":
    _self_check()
