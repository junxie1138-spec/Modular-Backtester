from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class SeasonalInnovationParams:
    season_window: int = 504
    slope_window: int = 6
    smooth_window: int = 3
    vol_window: int = 20
    min_samples: int = 40
    innovation_threshold: float = 0.6
    seasonal_threshold: float = 0.1
    holding_bars: int = 4


class GeneratedStrategy(BaseStrategy[SeasonalInnovationParams]):
    strategy_id = "gen_local-w2_1790991870"

    @classmethod
    def params_type(cls):
        return SeasonalInnovationParams

    @staticmethod
    def warmup_bars(params):
        return int(params.season_window + max(
            params.vol_window, params.slope_window + params.smooth_window
        ) + params.holding_bars + 2)

    def indicators(self, data, params):
        if (min(params.season_window, params.slope_window,
                params.smooth_window, params.min_samples) < 1
                or params.vol_window < 2 or not 3 <= params.holding_bars <= 5):
            raise ValueError("Invalid lookback or holding period")
        log_close = np.log(data["close"].where(data["close"] > 0))
        returns = log_close.diff()
        volatility = returns.rolling(params.vol_window).std(ddof=0)
        latent_slope = returns.rolling(params.slope_window).mean().rolling(
            params.smooth_window
        ).mean()
        innovation = latent_slope.diff().div(volatility.where(volatility > 1e-12))
        weekday = pd.Series(data.index.dayofweek, index=data.index)
        origin_weekday = weekday.shift(params.holding_bars)
        matured_return = log_close.diff(params.holding_bars)
        baseline = pd.Series(np.nan, index=data.index, dtype=float)
        spread = baseline.copy()
        expected_return = baseline.copy()
        samples = pd.Series(0.0, index=data.index)
        for day in range(5):
            # Baselines exclude today's innovation; outcomes are already matured.
            history = innovation.where(weekday.eq(day)).shift(1)
            count = history.notna().astype(float).rolling(
                params.season_window, min_periods=1
            ).sum()
            mean = history.fillna(0.0).rolling(
                params.season_window, min_periods=1
            ).sum().div(count.where(count > 0))
            second = history.pow(2).fillna(0.0).rolling(
                params.season_window, min_periods=1
            ).sum().div(count.where(count > 0))
            std = (second - mean.pow(2)).clip(lower=0.0).pow(0.5)
            outcomes = matured_return.where(origin_weekday.eq(day))
            outcome_count = outcomes.notna().astype(float).rolling(
                params.season_window, min_periods=1
            ).sum()
            outcome_mean = outcomes.fillna(0.0).rolling(
                params.season_window, min_periods=1
            ).sum().div(outcome_count.where(outcome_count > 0))
            current = weekday.eq(day)
            baseline.loc[current] = mean.loc[current]
            spread.loc[current] = std.loc[current]
            expected_return.loc[current] = outcome_mean.loc[current]
            samples.loc[current] = np.minimum(
                count.loc[current], outcome_count.loc[current]
            )
        innovation_z = (innovation - baseline).div(spread.where(spread > 1e-12))
        seasonal_score = expected_return.div(
            volatility.where(volatility > 1e-12) * np.sqrt(params.holding_bars)
        )
        ready = (
            samples.ge(params.min_samples)
            & innovation_z.notna() & seasonal_score.notna()
            & latent_slope.notna()
        )
        result = pd.DataFrame({
            "log_close": log_close, "returns": returns,
            "volatility": volatility, "latent_slope": latent_slope,
            "innovation": innovation, "weekday": weekday,
            "origin_weekday": origin_weekday, "matured_return": matured_return,
            "weekday_baseline": baseline, "weekday_spread": spread,
            "expected_return": expected_return, "samples": samples,
            "innovation_z": innovation_z, "seasonal_score": seasonal_score,
            "ready": ready,
        }, index=data.index)
        return result.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    def generate_signals(self, data, indicators, ctx, params):
        eligible = (
            indicators["ready"].astype(bool)
            & indicators["latent_slope"].gt(0.0)
            & indicators["innovation_z"].gt(params.innovation_threshold)
            & indicators["seasonal_score"].gt(params.seasonal_threshold)
        ).to_numpy()
        strength = (
            (indicators["innovation_z"] - params.innovation_threshold).clip(lower=0)
            * (indicators["seasonal_score"] - params.seasonal_threshold).clip(lower=0)
        ).clip(upper=50.0)
        entry_size = (0.2 + 0.75 * (1.0 - np.exp(-strength))).to_numpy()
        signal = np.zeros(len(data), dtype=int)
        size = np.full(len(data), 0.2, dtype=float)
        exit_bar = -1
        frozen_size = 0.2
        warmup = self.warmup_bars(params)
        for i in range(len(data)):
            if exit_bar >= 0:
                if i == exit_bar:
                    exit_bar = -1
                    continue
                signal[i] = 1
                size[i] = frozen_size
            elif i >= warmup and eligible[i]:
                exit_bar = i + params.holding_bars
                frozen_size = float(entry_size[i])
                signal[i] = 1
                size[i] = frozen_size
        df = pd.DataFrame({"signal": signal, "size": size}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = df["size"].shift(1).fillna(0.2).astype(float)
        return SignalFrame(data=df, signal_column="signal", size_column="size")
