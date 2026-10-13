from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class ForecastParams:
    slope_window: int = 15
    calibration_window: int = 100
    forecast_horizon: int = 18
    entry_threshold: float = 0.75
    ridge_fraction: float = 0.05


class GeneratedStrategy(BaseStrategy[ForecastParams]):
    strategy_id = "gen_local-w2_1790998061"

    @classmethod
    def params_type(cls) -> type[ForecastParams]:
        return ForecastParams

    def warmup_bars(self, params: ForecastParams) -> int:
        return int(max(3, params.slope_window) + max(10, params.calibration_window) + 3)

    def indicators(self, data: pd.DataFrame, params: ForecastParams) -> pd.DataFrame:
        window = max(3, int(params.slope_window))
        calibration = max(10, int(params.calibration_window))
        horizon = max(1, int(params.forecast_horizon))
        close = data["close"].astype(float)
        log_price = np.log(close.where(close > 0)).replace([np.inf, -np.inf], np.nan)
        returns = log_price.diff()

        # Causal least-squares slope, with newest observations weighted positively.
        slope = pd.Series(0.0, index=data.index)
        for lag in range(window):
            slope = slope + ((window - 1) / 2.0 - lag) * log_price.shift(lag)
        slope = slope / (window * (window * window - 1) / 12.0)
        innovation = slope.diff()
        lag_one = innovation.shift(1)
        lag_two = innovation.shift(2)

        m11 = (lag_one * lag_one).rolling(calibration).mean()
        m22 = (lag_two * lag_two).rolling(calibration).mean()
        m12 = (lag_one * lag_two).rolling(calibration).mean()
        b1 = (lag_one * innovation).rolling(calibration).mean()
        b2 = (lag_two * innovation).rolling(calibration).mean()
        ridge = max(0.0, float(params.ridge_fraction)) * (m11 + m22) / 2.0
        v11 = m11 + ridge
        v22 = m22 + ridge
        determinant = v11 * v22 - m12 * m12
        denominator = determinant.where(determinant > 0)
        ar_one = ((b1 * v22 - b2 * m12) / denominator).clip(-1.5, 1.5)
        ar_two = ((b2 * v11 - b1 * m12) / denominator).clip(-0.85, 0.85)
        # Keep coefficients inside the stationary AR(2) triangle.
        coefficient_limit = 0.98 - ar_two
        ar_one = ar_one.clip(lower=-coefficient_limit, upper=coefficient_limit)

        future_innovation = innovation.copy()
        previous_innovation = lag_one.copy()
        displacement = horizon * slope
        for step in range(1, horizon + 1):
            next_innovation = ar_one * future_innovation + ar_two * previous_innovation
            displacement = displacement + (horizon - step + 1) * next_innovation
            previous_innovation, future_innovation = future_innovation, next_innovation

        daily_volatility = returns.rolling(calibration).std()
        forecast_risk = daily_volatility * np.sqrt(float(horizon))
        forecast_score = displacement / forecast_risk.where(forecast_risk > 0)
        ready = np.isfinite(forecast_score) & np.isfinite(ar_one) & np.isfinite(ar_two)
        ready = ready & (np.arange(len(data)) >= self.warmup_bars(params) - 1)
        result = pd.DataFrame({
            "log_price": log_price,
            "log_return": returns,
            "latent_slope": slope,
            "slope_innovation": innovation,
            "innovation_lag_one": lag_one,
            "innovation_lag_two": lag_two,
            "innovation_ar_one": ar_one,
            "innovation_ar_two": ar_two,
            "daily_volatility": daily_volatility,
            "forecast_risk": forecast_risk,
            "forecast_displacement": displacement,
            "forecast_score": forecast_score,
            "ready": ready.astype(float),
        }, index=data.index)
        return result.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame,
                         ctx: StrategyContext, params: ForecastParams) -> SignalFrame:
        threshold = max(1e-9, float(params.entry_threshold))
        ready = indicators["ready"].eq(1.0)
        score = indicators["forecast_score"]
        candidates = np.where(ready & score.ge(threshold), 1,
                              np.where(ready & score.le(-threshold), -1, 0))
        signals = np.zeros(len(data), dtype=int)
        position = 0
        for bar, candidate in enumerate(candidates):
            if position == 0:
                position = int(candidate)
            elif candidate == -position:
                # The exact opposing entry predicate closes the position.
                # Re-entry requires a later close, preserving a flat exit bar.
                position = 0
            signals[bar] = position

        df = pd.DataFrame({"signal": signals, "size": 1.0}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")
