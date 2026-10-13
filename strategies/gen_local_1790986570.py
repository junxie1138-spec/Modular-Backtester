from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class AccelerationParams:
    correlation_window: int = 60
    atr_multiple: float = 3.0


class GeneratedStrategy(BaseStrategy[AccelerationParams]):
    strategy_id = "gen_local_1790986570"

    @classmethod
    def params_type(cls) -> type[AccelerationParams]:
        return AccelerationParams

    def warmup_bars(self, params: AccelerationParams) -> int:
        return int(max(15, params.correlation_window + 7))

    def indicators(self, data: pd.DataFrame, params: AccelerationParams) -> pd.DataFrame:
        window = int(params.correlation_window)
        close = data["close"].astype(float)
        roc = close.div(close.shift(5).replace(0.0, np.nan)).sub(1.0)
        acceleration = roc.diff()
        lagged = acceleration.shift(1)
        mean_acceleration = acceleration.rolling(window, min_periods=window).mean()
        mean_lagged = lagged.rolling(window, min_periods=window).mean()
        lagged_variance = lagged.rolling(window, min_periods=window).var()
        covariance = acceleration.rolling(window, min_periods=window).cov(lagged)
        beta = covariance.div(lagged_variance.replace(0.0, np.nan))
        forecast = mean_acceleration + beta * (acceleration - mean_lagged)
        acceleration_noise = acceleration.rolling(window, min_periods=window).std()
        forecast_strength = forecast.div(acceleration_noise.replace(0.0, np.nan))
        previous_close = close.shift(1)
        true_range = pd.concat([
            data["high"] - data["low"],
            (data["high"] - previous_close).abs(),
            (data["low"] - previous_close).abs(),
        ], axis=1).max(axis=1)
        atr = true_range.rolling(14, min_periods=14).mean()
        result = pd.DataFrame({
            "roc": roc,
            "acceleration": acceleration,
            "lagged_acceleration": lagged,
            "mean_acceleration": mean_acceleration,
            "mean_lagged_acceleration": mean_lagged,
            "lagged_variance": lagged_variance,
            "covariance": covariance,
            "beta": beta,
            "forecast": forecast,
            "acceleration_noise": acceleration_noise,
            "forecast_strength": forecast_strength,
            "previous_close": previous_close,
            "true_range": true_range,
            "atr": atr,
        }, index=data.index)
        return result.replace([np.inf, -np.inf], np.nan)

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame,
                         ctx: StrategyContext, params: AccelerationParams) -> SignalFrame:
        eligible = (
            (indicators["roc"] > 0.0)
            & (indicators["acceleration"] < 0.0)
            & (indicators["beta"] < -0.2)
            & (indicators["forecast_strength"] > 0.25)
            & (indicators["atr"] > 0.0)
        ).fillna(False).to_numpy(dtype=bool)
        opens = data["open"].to_numpy(dtype=float)
        lows = data["low"].to_numpy(dtype=float)
        closes = data["close"].to_numpy(dtype=float)
        atr = indicators["atr"].to_numpy(dtype=float)
        decisions = np.zeros(len(data), dtype=int)
        active = False
        armed = False
        entry_price = 0.0
        stop = 0.0
        peak_close = 0.0
        entry_bar = -1
        warmup = self.warmup_bars(params)

        for i in range(len(data)):
            # A previous close's entry decision fills at this bar's open.
            if not active and i > 0 and decisions[i - 1] == 1:
                if (np.isfinite(opens[i]) and opens[i] > 0.0
                        and np.isfinite(atr[i - 1]) and atr[i - 1] > 0.0):
                    active = True
                    armed = False
                    entry_price = opens[i]
                    entry_bar = i
                    peak_close = entry_price
                    stop = entry_price - params.atr_multiple * atr[i - 1]

            if active:
                # Test the existing stop before incorporating this close.
                stop_hit = np.isfinite(lows[i]) and lows[i] <= stop
                expired = i - entry_bar + 1 >= 18
                if stop_hit or expired:
                    active = False
                    decisions[i] = 0
                    continue

                if np.isfinite(closes[i]):
                    peak_close = max(peak_close, closes[i])
                    if closes[i] >= entry_price * 1.02:
                        armed = True
                    if armed:
                        stop = max(stop, entry_price)
                        if np.isfinite(atr[i]) and atr[i] > 0.0:
                            stop = max(stop, peak_close - params.atr_multiple * atr[i])
                    if closes[i] <= stop:
                        active = False
                        decisions[i] = 0
                        continue
                decisions[i] = 1
            elif i >= warmup - 1 and eligible[i]:
                decisions[i] = 1

        df = pd.DataFrame({"signal": decisions, "size": 1.0}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")
