from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class AutocorrelationParams:
    ma_length: int = 20
    correlation_length: int = 60
    atr_length: int = 14
    entry_z: float = 1.25
    forecast_threshold: float = 0.60
    minimum_persistence: float = 0.15
    maximum_persistence: float = 0.90
    reset_z: float = 0.35
    stop_atr: float = 1.70
    holding_bars: int = 4


class GeneratedStrategy(BaseStrategy[AutocorrelationParams]):
    strategy_id = "gen_local_1790997093"

    @classmethod
    def params_type(cls) -> type[AutocorrelationParams]:
        return AutocorrelationParams

    def warmup_bars(self, params: AutocorrelationParams) -> int:
        return int(max(params.ma_length + params.correlation_length,
                       params.atr_length + 1))

    def indicators(self, data: pd.DataFrame, params: AutocorrelationParams) -> pd.DataFrame:
        out = pd.DataFrame(index=data.index)
        close = data["close"].astype(float)
        out["ma"] = close.rolling(params.ma_length, min_periods=params.ma_length).mean()
        out["dispersion"] = close.rolling(params.ma_length, min_periods=params.ma_length).std(ddof=0)
        out["z"] = (close - out["ma"]) / out["dispersion"].replace(0.0, np.nan)
        out["lagged_z"] = out["z"].shift(1)
        length = params.correlation_length
        out["covariance"] = out["z"].rolling(length, min_periods=length).cov(out["lagged_z"], ddof=0)
        out["lag_variance"] = out["lagged_z"].rolling(length, min_periods=length).var(ddof=0)
        out["persistence"] = out["covariance"] / out["lag_variance"].replace(0.0, np.nan)
        out["z_mean"] = out["z"].rolling(length, min_periods=length).mean()
        out["lag_mean"] = out["lagged_z"].rolling(length, min_periods=length).mean()
        out["intercept"] = out["z_mean"] - out["persistence"] * out["lag_mean"]
        phi = out["persistence"].where(
            out["persistence"].between(params.minimum_persistence, params.maximum_persistence)
        )
        out["equilibrium_z"] = out["intercept"] / (1.0 - phi).replace(0.0, np.nan)
        horizon = min(5, max(3, int(params.holding_bars)))
        out["forecast_change"] = (out["equilibrium_z"] - out["z"]) * (1.0 - phi.pow(horizon))
        out["strength"] = out["forecast_change"].abs()
        out["previous_close"] = close.shift(1)
        out["true_range"] = pd.concat([
            data["high"] - data["low"],
            (data["high"] - out["previous_close"]).abs(),
            (data["low"] - out["previous_close"]).abs(),
        ], axis=1).max(axis=1)
        out["atr"] = out["true_range"].rolling(params.atr_length, min_periods=params.atr_length).mean()
        return out.replace([np.inf, -np.inf], np.nan)

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame,
                         ctx: StrategyContext, params: AutocorrelationParams) -> SignalFrame:
        count = len(data)
        signals = np.zeros(count, dtype=int)
        sizes = np.ones(count, dtype=float)
        closes = data["close"].to_numpy(dtype=float)
        opens = data["open"].to_numpy(dtype=float)
        zscores = indicators["z"].to_numpy(dtype=float)
        forecasts = indicators["forecast_change"].to_numpy(dtype=float)
        atrs = indicators["atr"].to_numpy(dtype=float)
        horizon = min(5, max(3, int(params.holding_bars)))
        threshold = max(float(params.forecast_threshold), 1e-9)
        warmup = self.warmup_bars(params)
        position = 0
        entry_index = -1
        entry_atr = np.nan
        stop_price = np.nan
        trade_size = 1.0
        armed = True

        for i in range(count):
            if position != 0:
                if i == entry_index:
                    # The prior close's order fills at this bar's open.
                    if not np.isfinite(opens[i]) or opens[i] <= 0.0:
                        position = 0
                        continue
                    # Freeze both the actual entry price and the decision-time ATR.
                    stop_price = opens[i] - position * params.stop_atr * entry_atr
                stopped = (
                    np.isfinite(closes[i])
                    and position * (closes[i] - stop_price) < 0.0
                )
                expired = i - entry_index + 1 >= horizon
                if stopped or expired:
                    # This close's flat decision executes on the following bar.
                    position = 0
                    continue
                signals[i] = position
                sizes[i] = trade_size
                continue

            if np.isfinite(zscores[i]) and abs(zscores[i]) <= params.reset_z:
                armed = True
            if i < warmup - 1 or not armed:
                continue
            if not (np.isfinite(zscores[i]) and np.isfinite(forecasts[i])
                    and np.isfinite(atrs[i]) and atrs[i] > 0.0
                    and np.isfinite(closes[i]) and closes[i] > 0.0):
                continue
            strength = abs(forecasts[i])
            if abs(zscores[i]) < params.entry_z or strength < threshold:
                continue
            # Require the forecast to point back toward the moving average.
            if zscores[i] * forecasts[i] >= 0.0:
                continue
            position = 1 if forecasts[i] > 0.0 else -1
            trade_size = float(np.clip(
                0.25 + 0.70 * (strength - threshold) / (2.0 * threshold),
                0.25, 0.95
            ))
            entry_index = i + 1
            entry_atr = atrs[i]
            stop_price = np.nan
            armed = False
            signals[i] = position
            sizes[i] = trade_size

        df = pd.DataFrame({"signal": signals, "size": sizes}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = df["size"].shift(1).fillna(1.0).astype(float)
        return SignalFrame(data=df, signal_column="signal", size_column="size")
