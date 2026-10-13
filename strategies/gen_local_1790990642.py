from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class ChangepointParams:
    fractional_order: float = 0.45
    fractional_window: int = 32
    recent_window: int = 3
    baseline_window: int = 12
    percentile_window: int = 126
    entry_percentile: float = 0.90
    atr_window: int = 14
    trailing_atr: float = 1.5
    holding_bars: int = 4


class GeneratedStrategy(BaseStrategy[ChangepointParams]):
    strategy_id = "gen_local_1790990642"

    @classmethod
    def params_type(cls) -> type[ChangepointParams]:
        return ChangepointParams

    @staticmethod
    def warmup_bars(params: ChangepointParams) -> int:
        return int(max(
            params.fractional_window + params.recent_window
            + params.baseline_window + params.percentile_window + 2,
            params.atr_window + 1,
        ))

    @staticmethod
    def indicators(data: pd.DataFrame, params: ChangepointParams) -> pd.DataFrame:
        close = data["close"].astype(float)
        weights = np.ones(params.fractional_window, dtype=float)
        for lag in range(1, params.fractional_window):
            weights[lag] = (
                -weights[lag - 1]
                * (params.fractional_order - lag + 1) / lag
            )
        fractional_close = pd.Series(0.0, index=data.index)
        for lag, weight in enumerate(weights):
            fractional_close = fractional_close + weight * close.shift(lag)

        recent_mean = fractional_close.rolling(
            params.recent_window, min_periods=params.recent_window
        ).mean()
        preceding = fractional_close.shift(params.recent_window)
        baseline_mean = preceding.rolling(
            params.baseline_window, min_periods=params.baseline_window
        ).mean()
        baseline_scale = preceding.rolling(
            params.baseline_window, min_periods=params.baseline_window
        ).std(ddof=0).replace(0.0, np.nan)
        changepoint = ((recent_mean - baseline_mean) / baseline_scale).replace(
            [np.inf, -np.inf], np.nan
        )
        threshold = changepoint.abs().shift(1).rolling(
            params.percentile_window, min_periods=params.percentile_window
        ).quantile(params.entry_percentile)

        previous_close = close.shift(1)
        true_range = pd.concat([
            data["high"] - data["low"],
            (data["high"] - previous_close).abs(),
            (data["low"] - previous_close).abs(),
        ], axis=1).max(axis=1)
        atr = true_range.rolling(
            params.atr_window, min_periods=params.atr_window
        ).mean().where(lambda values: values > 0.0)

        return pd.DataFrame({
            "fractional_close": fractional_close,
            "fractional_change": fractional_close.diff(),
            "recent_mean": recent_mean,
            "baseline_mean": baseline_mean,
            "baseline_scale": baseline_scale,
            "changepoint": changepoint,
            "entry_threshold": threshold,
            "true_range": true_range,
            "atr": atr,
        }, index=data.index).replace([np.inf, -np.inf], np.nan)

    @staticmethod
    def generate_signals(
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: ChangepointParams,
    ) -> SignalFrame:
        score = indicators["changepoint"]
        threshold = indicators["entry_threshold"]
        entries = (
            score.gt(threshold)
            & score.gt(0.0)
            & score.shift(1).le(threshold.shift(1))
            & indicators["fractional_change"].gt(0.0)
            & indicators["atr"].notna()
        ).fillna(False).to_numpy(dtype=bool)
        closes = data["close"].to_numpy(dtype=float)
        atrs = indicators["atr"].to_numpy(dtype=float)
        signals = np.zeros(len(data), dtype=int)
        active = False
        age = 0
        high_water = np.nan
        stop = -np.inf

        for i in range(len(data)):
            close = closes[i]
            atr = atrs[i]
            valid = np.isfinite(close) and np.isfinite(atr) and atr > 0.0
            if active:
                age += 1
                if valid:
                    # Entry fills on the next bar; start tracking its closing high there.
                    high_water = max(high_water, close) if np.isfinite(high_water) else close
                    stop = max(stop, high_water - params.trailing_atr * atr)
                    if close <= stop or age >= params.holding_bars:
                        active = False
                else:
                    active = False
            elif entries[i] and valid:
                active = True
                age = 0
                high_water = np.nan
                stop = -np.inf
            signals[i] = int(active)

        df = pd.DataFrame({"signal": signals, "size": 0.95}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")
