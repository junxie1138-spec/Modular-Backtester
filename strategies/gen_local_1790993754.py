from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class AnalogParams:
    slope_window: int = 12
    latent_window: int = 4
    volatility_window: int = 20
    shape_spacing: int = 4
    archive_window: int = 252
    neighbors: int = 12
    forecast_horizon: int = 18
    minimum_forecast: float = 0.012
    maximum_distance: float = 1.25
    profit_target: float = 0.04
    maximum_holding: int = 18
    minimum_size: float = 0.25
    maximum_size: float = 0.95


class GeneratedStrategy(BaseStrategy[AnalogParams]):
    strategy_id = 'gen_local_1790993754'

    @classmethod
    def params_type(cls):
        return AnalogParams

    def warmup_bars(self, params: AnalogParams) -> int:
        return int(params.slope_window + params.latent_window + params.volatility_window + 2 * params.shape_spacing + params.archive_window + params.forecast_horizon + 1)

    def indicators(self, data: pd.DataFrame, params: AnalogParams) -> pd.DataFrame:
        out = pd.DataFrame(index=data.index)
        out['log_close'] = np.log(data['close'].where(data['close'] > 0))
        out['log_return'] = out['log_close'].diff()
        out['volatility'] = out['log_return'].rolling(params.volatility_window, min_periods=params.volatility_window).std().replace(0.0, np.nan)
        n = params.slope_window
        center = (n - 1) / 2.0
        denominator = n * (n * n - 1) / 12.0
        numerator = pd.Series(0.0, index=data.index)
        for lag in range(n):
            numerator = numerator + (center - lag) * out['log_close'].shift(lag)
        out['slope'] = numerator / denominator
        out['latent_slope'] = out['slope'].rolling(params.latent_window, min_periods=params.latent_window).mean()
        out['innovation'] = (out['slope'] - out['latent_slope'].shift(1)) / out['volatility']
        out['shape_0'] = out['innovation']
        out['shape_1'] = out['innovation'].shift(params.shape_spacing)
        out['shape_2'] = out['innovation'].shift(2 * params.shape_spacing)
        out['matured_return'] = np.expm1(out['log_close'].diff(params.forecast_horizon))

        features = out[['shape_0', 'shape_1', 'shape_2']].to_numpy(dtype=float)
        outcomes = out['matured_return'].to_numpy(dtype=float)
        forecasts = np.full(len(data), np.nan)
        distances = np.full(len(data), np.nan)
        counts = np.zeros(len(data), dtype=int)
        horizon = params.forecast_horizon
        separation = 2 * params.shape_spacing
        for t in range(len(data)):
            if not np.isfinite(features[t]).all():
                continue
            end = t - horizon - separation
            start = end - params.archive_window + 1
            if start < 0:
                continue
            candidates = np.arange(start, end + 1)
            historical = features[candidates]
            realized = outcomes[candidates + horizon]
            valid = np.isfinite(historical).all(axis=1) & np.isfinite(realized)
            if valid.sum() < params.neighbors:
                continue
            historical = historical[valid]
            realized = realized[valid]
            distance = np.sqrt(np.mean((historical - features[t]) ** 2, axis=1))
            nearest = np.argsort(distance, kind='stable')[:params.neighbors]
            selected_distance = distance[nearest]
            weights = np.exp(-0.5 * (selected_distance / max(params.maximum_distance, 1e-8)) ** 2)
            weight_sum = weights.sum()
            if weight_sum <= 1e-12:
                continue
            forecasts[t] = np.dot(weights, realized[nearest]) / weight_sum
            distances[t] = selected_distance.mean()
            counts[t] = len(nearest)

        out['analog_forecast'] = forecasts
        out['analog_distance'] = distances
        out['analog_count'] = counts
        out['forecast_strength'] = out['analog_forecast'] / (out['volatility'] * np.sqrt(horizon))
        out['similarity_strength'] = (1.0 - out['analog_distance'] / max(params.maximum_distance, 1e-8)).clip(0.0, 1.0)
        out['conviction'] = (out['forecast_strength'].clip(0.0, 1.0) * (0.5 + 0.5 * out['similarity_strength'])).fillna(0.0)
        out['entry_ready'] = ((out['analog_forecast'] >= params.minimum_forecast) & (out['analog_distance'] <= params.maximum_distance) & (out['innovation'] > 0.0) & (out['latent_slope'] > 0.0) & (out['analog_count'] >= params.neighbors)).astype(int)
        return out.replace([np.inf, -np.inf], np.nan)

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: AnalogParams) -> SignalFrame:
        df = pd.DataFrame(index=data.index)
        signals = np.zeros(len(data), dtype=int)
        sizes = np.full(len(data), float(params.minimum_size))
        ready = indicators['entry_ready'].fillna(0).to_numpy(dtype=int)
        conviction = indicators['conviction'].fillna(0.0).to_numpy(dtype=float)
        opens = data['open'].to_numpy(dtype=float)
        closes = data['close'].to_numpy(dtype=float)
        active = False
        entry_price = np.nan
        entry_bar = -1
        held_size = float(params.minimum_size)
        first_bar = self.warmup_bars(params) - 1

        for i in range(len(data)):
            exited = False
            if active:
                if i == entry_bar:
                    entry_price = opens[i]
                valid_price = np.isfinite(entry_price) and entry_price > 0.0
                target_hit = valid_price and np.isfinite(closes[i]) and closes[i] >= entry_price * (1.0 + params.profit_target)
                timed_out = i - entry_bar + 1 >= params.maximum_holding
                invalid_fill = i == entry_bar and not valid_price
                if target_hit or timed_out or invalid_fill:
                    active = False
                    entry_price = np.nan
                    exited = True
            if not active and not exited and i >= first_bar and ready[i] == 1:
                active = True
                entry_bar = i + 1
                entry_price = np.nan
                held_size = float(np.clip(params.minimum_size + (params.maximum_size - params.minimum_size) * conviction[i], params.minimum_size, params.maximum_size))
            signals[i] = int(active)
            sizes[i] = held_size if active or exited else float(params.minimum_size)

        df['signal'] = signals
        df['size'] = sizes
        df['signal'] = df['signal'].shift(1).fillna(0).astype(int)
        df['size'] = df['size'].shift(1).fillna(float(params.minimum_size)).astype(float)
        return SignalFrame(data=df, signal_column='signal', size_column='size')
