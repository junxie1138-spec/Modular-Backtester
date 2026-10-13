from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class StrategyParams:
    range_window: int = 12
    baseline_window: int = 60
    atr_window: int = 14
    drift_window: int = 20
    turbulent_dispersion: float = 0.75
    stable_dispersion: float = 0.40
    turbulent_range_ratio: float = 1.10
    stable_range_floor: float = 0.85
    min_signal_to_noise: float = 0.65
    regime_memory: int = 40
    hold_bars: int = 18
    stop_atr: float = 3.0
    position_size: float = 0.95


class GeneratedStrategy(BaseStrategy[StrategyParams]):
    strategy_id = "gen_local_1791207770"

    @classmethod
    def params_type(cls) -> type[StrategyParams]:
        return StrategyParams

    def warmup_bars(self, params: StrategyParams) -> int:
        return int(max(params.range_window, params.baseline_window,
                       params.atr_window, params.drift_window) + 1)

    def indicators(self, data: pd.DataFrame, params: StrategyParams) -> pd.DataFrame:
        out = pd.DataFrame(index=data.index)
        out["previous_close"] = data["close"].shift(1)
        out["true_range"] = pd.concat([
            data["high"] - data["low"],
            (data["high"] - out["previous_close"]).abs(),
            (data["low"] - out["previous_close"]).abs(),
        ], axis=1).max(axis=1, skipna=False)
        out["normalized_range"] = out["true_range"] / out["previous_close"].where(out["previous_close"] > 0)
        out["range_mean"] = out["normalized_range"].rolling(params.range_window, min_periods=params.range_window).mean()
        out["range_std"] = out["normalized_range"].rolling(params.range_window, min_periods=params.range_window).std(ddof=0)
        out["range_baseline"] = out["normalized_range"].rolling(params.baseline_window, min_periods=params.baseline_window).mean()
        out["dispersion"] = out["range_std"] / out["range_mean"].where(out["range_mean"] > 0)
        out["range_ratio"] = out["range_mean"] / out["range_baseline"].where(out["range_baseline"] > 0)
        out["atr"] = out["true_range"].rolling(params.atr_window, min_periods=params.atr_window).mean()
        out["log_return"] = np.log(data["close"].where(data["close"] > 0)).diff()
        out["drift"] = out["log_return"].rolling(params.drift_window, min_periods=params.drift_window).sum()
        out["return_energy"] = out["log_return"].pow(2).rolling(params.drift_window, min_periods=params.drift_window).sum()
        out["noise"] = np.sqrt(out["return_energy"])
        out["signal_to_noise"] = out["drift"] / out["noise"].where(out["noise"] > 0)
        return out.replace([np.inf, -np.inf], np.nan)

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame,
                         ctx: StrategyContext, params: StrategyParams) -> SignalFrame:
        df = pd.DataFrame(index=data.index)
        signals = np.zeros(len(data), dtype=int)
        closes = data["close"].to_numpy(dtype=float)
        values = indicators[["dispersion", "range_ratio", "signal_to_noise", "atr"]].to_numpy(dtype=float)
        armed_at = -1
        confirmations = 0
        entry_bar = -1
        high_water = np.nan
        trailing_stop = np.nan

        for i in range(len(data)):
            close = closes[i]
            dispersion, range_ratio, signal_to_noise, atr = values[i]
            valid = np.isfinite(close) and close > 0 and np.isfinite(values[i]).all() and atr > 0

            if entry_bar >= 0:
                if np.isfinite(close) and close > 0:
                    high_water = max(high_water, close)
                    if np.isfinite(atr) and atr > 0:
                        trailing_stop = max(trailing_stop, high_water - params.stop_atr * atr)
                    stop_hit = close <= trailing_stop
                else:
                    stop_hit = False
                if stop_hit or i - entry_bar >= params.hold_bars:
                    entry_bar = -1
                    armed_at = -1
                    confirmations = 0
                else:
                    signals[i] = 1
                continue

            if not valid:
                confirmations = 0
                continue

            if dispersion >= params.turbulent_dispersion and range_ratio >= params.turbulent_range_ratio:
                armed_at = i
                confirmations = 0

            if armed_at >= 0 and i - armed_at > params.regime_memory:
                armed_at = -1
                confirmations = 0

            qualifying = (
                armed_at >= 0
                and i > armed_at
                and dispersion <= params.stable_dispersion
                and range_ratio >= params.stable_range_floor
                and signal_to_noise >= params.min_signal_to_noise
            )
            confirmations = confirmations + 1 if qualifying else 0
            if confirmations >= 2:
                entry_bar = i
                high_water = close
                trailing_stop = close - params.stop_atr * atr
                armed_at = -1
                confirmations = 0
                signals[i] = 1

        df["signal"] = signals
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = float(params.position_size)
        return SignalFrame(data=df, signal_column="signal", size_column="size")
