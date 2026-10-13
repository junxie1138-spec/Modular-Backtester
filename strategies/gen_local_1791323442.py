from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class RangeMigrationParams:
    fast_window: int = 8
    slow_window: int = 60
    atr_window: int = 14
    volatility_window: int = 20
    contraction_threshold: float = 0.85
    overlap_threshold: float = 0.55
    drift_threshold: float = 0.03
    target_volatility: float = 0.12
    trail_atr: float = 3.5
    holding_bars: int = 20
    max_size: float = 0.95


class GeneratedStrategy(BaseStrategy[RangeMigrationParams]):
    strategy_id = "gen_local_1791323442"

    @classmethod
    def params_type(cls) -> type[RangeMigrationParams]:
        return RangeMigrationParams

    def warmup_bars(self, params: RangeMigrationParams) -> int:
        return int(max(params.fast_window, params.slow_window,
                       params.atr_window, params.volatility_window) + 2)

    def indicators(self, data: pd.DataFrame, params: RangeMigrationParams) -> pd.DataFrame:
        windows = (params.fast_window, params.slow_window,
                   params.atr_window, params.volatility_window)
        if any(w < 2 for w in windows) or params.holding_bars < 1:
            raise ValueError("Lookbacks must be at least two; holding_bars must be positive.")
        values = (params.contraction_threshold, params.overlap_threshold,
                  params.drift_threshold, params.target_volatility,
                  params.trail_atr, params.max_size)
        if not all(np.isfinite(v) for v in values):
            raise ValueError("Parameters must be finite.")
        if not (0 < params.contraction_threshold <= 1
                and 0 <= params.overlap_threshold <= 1
                and params.drift_threshold >= 0
                and params.target_volatility > 0
                and params.trail_atr > 0
                and 0.01 <= params.max_size <= 1):
            raise ValueError("Invalid thresholds, volatility target, trailing distance or size.")

        out = pd.DataFrame(index=data.index)
        high = data["high"].astype(float)
        low = data["low"].astype(float)
        close = data["close"].astype(float)
        valid = (np.isfinite(high) & np.isfinite(low) & np.isfinite(close)
                 & (low > 0) & (close > 0) & (high >= low)
                 & (close >= low) & (close <= high))
        high = high.where(valid)
        low = low.where(valid)
        close = close.where(valid)
        out["range"] = high - low
        out["relative_range"] = out["range"] / close
        out["fast_range"] = out["relative_range"].rolling(
            params.fast_window, min_periods=params.fast_window).mean()
        out["slow_range"] = out["relative_range"].rolling(
            params.slow_window, min_periods=params.slow_window).mean()
        out["contraction"] = out["fast_range"] / out["slow_range"].replace(0, np.nan)
        out["intersection"] = (np.minimum(high, high.shift(1))
                               - np.maximum(low, low.shift(1))).clip(lower=0)
        out["overlap"] = out["intersection"] / np.minimum(
            out["range"], out["range"].shift(1)).replace(0, np.nan)
        out["mean_overlap"] = out["overlap"].rolling(
            params.fast_window, min_periods=params.fast_window).mean()
        out["midpoint"] = (high + low) / 2.0
        out["midpoint_change"] = out["midpoint"].diff()
        out["mean_midpoint_change"] = out["midpoint_change"].rolling(
            params.fast_window, min_periods=params.fast_window).mean()
        out["mean_range"] = out["range"].rolling(
            params.fast_window, min_periods=params.fast_window).mean()
        out["migration"] = out["mean_midpoint_change"] / out["mean_range"].replace(0, np.nan)
        previous_close = close.shift(1)
        out["true_range"] = pd.concat(
            [out["range"], (high - previous_close).abs(),
             (low - previous_close).abs()], axis=1).max(axis=1).where(valid)
        out["atr"] = out["true_range"].rolling(
            params.atr_window, min_periods=params.atr_window).mean()
        out["log_range"] = np.log(high / low)
        out["range_variance"] = out["log_range"] ** 2 / (4.0 * np.log(2.0))
        out["annual_volatility"] = np.sqrt(252.0 * out["range_variance"].rolling(
            params.volatility_window, min_periods=params.volatility_window).mean())
        out = out.replace([np.inf, -np.inf], np.nan)
        out["entry_ok"] = (
            valid
            & (out["contraction"] <= params.contraction_threshold)
            & (out["mean_overlap"] >= params.overlap_threshold)
            & (out["migration"] >= params.drift_threshold)
            & (out["atr"] > 0)
            & (out["annual_volatility"] > 0)
        ).fillna(False)
        return out

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame,
                         ctx: StrategyContext, params: RangeMigrationParams) -> SignalFrame:
        close = data["close"].to_numpy(dtype=float)
        atr = indicators["atr"].to_numpy(dtype=float)
        volatility = indicators["annual_volatility"].to_numpy(dtype=float)
        qualifying = indicators["entry_ok"].fillna(False).to_numpy(dtype=bool)
        valid = (np.isfinite(close) & (close > 0)
                 & np.isfinite(atr) & (atr > 0)
                 & np.isfinite(volatility) & (volatility > 0))
        signal = np.zeros(len(data), dtype=int)
        active = False
        confirmation = 0
        entry_bar = -1
        high_water = np.nan
        stop = -np.inf

        for i in range(len(data)):
            if active:
                if not valid[i]:
                    active = False
                    confirmation = 0
                    high_water = np.nan
                    stop = -np.inf
                    continue
                # The first held close is on the bar after the entry decision.
                high_water = close[i] if np.isnan(high_water) else max(high_water, close[i])
                stop = max(stop, high_water - params.trail_atr * atr[i])
                if close[i] <= stop or i - entry_bar >= params.holding_bars:
                    active = False
                    confirmation = 0
                    high_water = np.nan
                    stop = -np.inf
                else:
                    signal[i] = 1
            else:
                confirmation = confirmation + 1 if qualifying[i] and valid[i] else 0
                if confirmation >= 2:
                    active = True
                    entry_bar = i
                    high_water = np.nan
                    stop = -np.inf
                    confirmation = 0
                    signal[i] = 1

        df = pd.DataFrame(index=data.index)
        df["signal"] = signal
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        exposure = (params.target_volatility
                    / indicators["annual_volatility"].replace(0, np.nan))
        exposure = exposure.replace([np.inf, -np.inf], np.nan).clip(
            lower=0.01, upper=params.max_size)
        df["size"] = exposure.shift(1).fillna(params.max_size).astype(float)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


if __name__ == "__main__":
    strategy = GeneratedStrategy()
    params = RangeMigrationParams(trail_atr=2.5)
    index = pd.date_range("2020-01-01", periods=6)
    data = pd.DataFrame({"close": [100., 100., 104., 101., 100., 100.]}, index=index)
    derived = pd.DataFrame({
        "atr": [1., 1., 1., 10., 1., 1.],
        "annual_volatility": [0.2] * 6,
        "entry_ok": [True, True, False, False, False, False],
    }, index=index)
    result = strategy.generate_signals(data, derived, None, params).data
    assert result["signal"].tolist() == [0, 0, 1, 1, 0, 0]
    assert (result["size"] > 0).all()
    data["close"] = 100.0
    derived["atr"] = 1.0
    derived["entry_ok"] = True
    params.holding_bars = 2
    result = strategy.generate_signals(data, derived, None, params).data
    assert result["signal"].tolist() == [0, 0, 1, 1, 0, 0]
    derived["entry_ok"] = [True, False, True, False, True, False]
    assert not strategy.generate_signals(data, derived, None, params).data["signal"].any()
