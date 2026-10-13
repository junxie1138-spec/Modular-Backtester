from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class RecoveryParams:
    slope_window: int = 8
    peak_window: int = 40
    volatility_window: int = 20
    volume_window: int = 20
    impulse_window: int = 2
    innovation_threshold: float = 0.15
    minimum_drawdown: float = 0.01
    attenuation_ratio: float = 0.50
    hold_bars: int = 4


class GeneratedStrategy(BaseStrategy[RecoveryParams]):
    strategy_id = "gen_local-w2_1791306510"

    @classmethod
    def params_type(cls) -> type[RecoveryParams]:
        return RecoveryParams

    def warmup_bars(self, params: RecoveryParams) -> int:
        return int(max(
            params.peak_window,
            params.slope_window + 3,
            params.volatility_window + 1,
            params.volume_window + 2 * params.impulse_window,
        ))

    def indicators(self, data: pd.DataFrame, params: RecoveryParams) -> pd.DataFrame:
        if min(params.slope_window, params.peak_window,
               params.volatility_window, params.volume_window,
               params.impulse_window) < 2:
            raise ValueError("Indicator windows must be at least two bars")
        if not 3 <= params.hold_bars <= 5:
            raise ValueError("hold_bars must be between three and five")
        if params.innovation_threshold < 0 or params.minimum_drawdown < 0:
            raise ValueError("Recovery thresholds must be nonnegative")
        if not 0 < params.attenuation_ratio < 1:
            raise ValueError("attenuation_ratio must be between zero and one")

        out = pd.DataFrame(index=data.index)
        out["log_close"] = np.log(data["close"].where(data["close"] > 0))
        out["log_open"] = np.log(data["open"].where(data["open"] > 0))
        out["log_return"] = out["log_close"].diff()
        out["return_volatility"] = out["log_return"].rolling(
            params.volatility_window, min_periods=params.volatility_window
        ).std(ddof=0)

        n = params.slope_window
        out["bar_position"] = np.arange(len(data), dtype=float)
        out["position_price"] = out["bar_position"] * out["log_close"]
        out["price_sum"] = out["log_close"].rolling(n, min_periods=n).sum()
        out["position_sum"] = out["bar_position"].rolling(n, min_periods=n).sum()
        out["position_price_sum"] = out["position_price"].rolling(n, min_periods=n).sum()
        out["latent_slope"] = (
            out["position_price_sum"]
            - out["position_sum"] * out["price_sum"] / n
        ) / (n * (n * n - 1) / 12.0)
        out["prior_slope"] = out["latent_slope"].shift(3)
        out["slope_innovation"] = out["latent_slope"] - out["prior_slope"]
        out["scaled_innovation"] = out["slope_innovation"] / (
            out["return_volatility"].replace(0.0, np.nan) / np.sqrt(n)
        )

        out["recent_peak"] = data["close"].rolling(
            params.peak_window, min_periods=params.peak_window
        ).max()
        out["drawdown"] = data["close"] / out["recent_peak"].replace(0.0, np.nan) - 1.0
        out["average_volume"] = data["volume"].rolling(
            params.volume_window, min_periods=params.volume_window
        ).mean()
        out["relative_volume"] = data["volume"].where(data["volume"] >= 0) / (
            out["average_volume"].where(out["average_volume"] > 0)
        )
        out["bearish_body"] = (out["log_open"] - out["log_close"]).clip(lower=0.0)
        out["selling_impulse"] = out["bearish_body"] * out["relative_volume"]
        out["recent_impulse"] = out["selling_impulse"].rolling(
            params.impulse_window, min_periods=params.impulse_window
        ).mean()
        out["previous_impulse"] = out["recent_impulse"].shift(params.impulse_window)
        out["close_change"] = data["close"].diff()
        return out.replace([np.inf, -np.inf], np.nan)

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: RecoveryParams,
    ) -> SignalFrame:
        slope_agrees = (
            (indicators["prior_slope"] < 0.0)
            & (indicators["latent_slope"] > 0.0)
            & (indicators["scaled_innovation"] >= params.innovation_threshold)
        )
        pressure_agrees = (
            (indicators["previous_impulse"] > 0.0)
            & (indicators["recent_impulse"] <=
               params.attenuation_ratio * indicators["previous_impulse"])
            & (indicators["close_change"] > 0.0)
        )
        required = ["prior_slope", "latent_slope", "scaled_innovation",
                    "drawdown", "recent_impulse", "previous_impulse", "close_change"]
        ready = indicators[required].notna().all(axis=1)
        ready &= np.arange(len(data)) >= self.warmup_bars(params) - 1
        entries = (
            ready & slope_agrees & pressure_agrees
            & (indicators["drawdown"] <= -params.minimum_drawdown)
        ).fillna(False).to_numpy(dtype=bool)

        signals = np.zeros(len(data), dtype=int)
        exit_bar = -1
        for i in range(len(data)):
            if exit_bar >= 0:
                if i < exit_bar:
                    signals[i] = 1
                else:
                    exit_bar = -1
                continue
            if entries[i]:
                signals[i] = 1
                exit_bar = i + params.hold_bars

        df = pd.DataFrame(index=data.index)
        df["signal"] = signals
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = 0.95
        return SignalFrame(data=df, signal_column="signal", size_column="size")
