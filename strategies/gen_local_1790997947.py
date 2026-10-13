from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class StrategyParams:
    trend_window: int = 24
    innovation_window: int = 3
    innovation_threshold: float = 0.08
    atr_window: int = 14
    volume_window: int = 20
    participation_window: int = 12
    minimum_group_bars: int = 3
    trailing_atr: float = 3.0
    max_hold_bars: int = 18


class GeneratedStrategy(BaseStrategy[StrategyParams]):
    strategy_id = "gen_local_1790997947"

    @classmethod
    def params_type(cls) -> type[StrategyParams]:
        return StrategyParams

    def warmup_bars(self, params: StrategyParams) -> int:
        return int(max(
            params.trend_window + params.innovation_window,
            params.atr_window + params.participation_window + 1,
            params.volume_window + params.participation_window + 1,
        ))

    def indicators(self, data: pd.DataFrame, params: StrategyParams) -> pd.DataFrame:
        out = pd.DataFrame(index=data.index)
        close = data["close"].astype(float).where(data["close"] > 0)
        out["log_close"] = np.log(close)
        out["log_return"] = out["log_close"].diff()
        previous_close = close.shift(1)
        out["true_range"] = pd.concat([
            data["high"] - data["low"],
            (data["high"] - previous_close).abs(),
            (data["low"] - previous_close).abs(),
        ], axis=1).max(axis=1)
        out["atr"] = out["true_range"].rolling(params.atr_window).mean()
        out["relative_atr"] = out["atr"].div(previous_close).where(lambda s: s > 0)

        # Fit log prices against time using rolling sufficient statistics.
        n = params.trend_window
        time = pd.Series(np.arange(len(data), dtype=float), index=data.index)
        time_mean = time.rolling(n).mean()
        out["trend_level"] = out["log_close"].rolling(n).mean()
        covariance = (time * out["log_close"]).rolling(n).mean() - time_mean * out["trend_level"]
        variance = (n * n - 1.0) / 12.0
        out["latent_slope"] = covariance / variance
        out["fitted_close"] = out["trend_level"] + out["latent_slope"] * (n - 1.0) / 2.0
        out["prior_forecast"] = (out["fitted_close"] + out["latent_slope"]).shift(1)
        out["trend_innovation"] = (out["log_close"] - out["prior_forecast"]) / out["relative_atr"]
        out["innovation_mean"] = out["trend_innovation"].rolling(params.innovation_window).mean()

        # Independently measure whether advances survive reduced participation.
        volume = data["volume"].astype(float).where(data["volume"] > 0)
        out["prior_volume_mean"] = volume.rolling(params.volume_window).mean().shift(1)
        valid = volume.notna() & out["prior_volume_mean"].notna() & out["log_return"].notna() & out["relative_atr"].notna()
        quiet = valid & volume.le(out["prior_volume_mean"])
        crowded = valid & volume.gt(out["prior_volume_mean"])
        out["quiet_session"] = quiet.astype(float)
        out["crowded_session"] = crowded.astype(float)
        out["normalized_return"] = out["log_return"] / out["relative_atr"]
        w = params.participation_window
        out["quiet_count"] = out["quiet_session"].rolling(w).sum()
        out["crowded_count"] = out["crowded_session"].rolling(w).sum()
        quiet_sum = out["normalized_return"].where(quiet, 0.0).rolling(w).sum()
        crowded_sum = out["normalized_return"].where(crowded, 0.0).rolling(w).sum()
        out["quiet_return"] = quiet_sum / out["quiet_count"].replace(0.0, np.nan)
        out["crowded_return"] = crowded_sum / out["crowded_count"].replace(0.0, np.nan)
        return out.replace([np.inf, -np.inf], np.nan)

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: StrategyParams) -> SignalFrame:
        trend_agrees = (
            indicators["latent_slope"].gt(0.0)
            & indicators["innovation_mean"].gt(params.innovation_threshold)
        )
        participation_agrees = (
            indicators["quiet_count"].ge(params.minimum_group_bars)
            & indicators["crowded_count"].ge(params.minimum_group_bars)
            & indicators["quiet_return"].gt(0.0)
            & indicators["quiet_return"].gt(indicators["crowded_return"])
        )
        entries = (trend_agrees & participation_agrees).fillna(False).to_numpy(dtype=bool)
        closes = data["close"].to_numpy(dtype=float)
        atrs = indicators["atr"].to_numpy(dtype=float)
        signals = np.zeros(len(data), dtype=int)
        active = False
        entry_bar = -1
        high_water = np.nan
        stop = np.nan
        warmup = self.warmup_bars(params)

        for i in range(len(data)):
            close = closes[i]
            atr = atrs[i]
            valid_close = np.isfinite(close) and close > 0.0
            valid_atr = np.isfinite(atr) and atr > 0.0
            if active:
                if valid_close:
                    high_water = max(high_water, close)
                    if valid_atr:
                        stop = max(stop, high_water - params.trailing_atr * atr)
                    if close <= stop:
                        active = False
                if i - entry_bar >= params.max_hold_bars:
                    active = False
            elif i >= warmup - 1 and entries[i] and valid_close and valid_atr:
                active = True
                entry_bar = i
                high_water = close
                stop = close - params.trailing_atr * atr
            signals[i] = int(active)

        df = pd.DataFrame(index=data.index)
        df["signal"] = signals
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = 1.0
        return SignalFrame(data=df, signal_column="signal", size_column="size")
