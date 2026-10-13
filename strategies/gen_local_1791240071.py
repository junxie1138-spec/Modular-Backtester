from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class AnalogParams:
    roc_days: int = 3
    shape_days: int = 5
    archive_days: int = 252
    neighbors: int = 5
    horizon: int = 8
    volatility_days: int = 20
    atr_days: int = 14
    max_distance: float = 0.6
    forecast_threshold: float = 0.005
    stop_atr: float = 2.0


class GeneratedStrategy(BaseStrategy[AnalogParams]):
    strategy_id = 'gen_local_1791240071'

    @classmethod
    def params_type(cls):
        return AnalogParams

    @staticmethod
    def warmup_bars(params: AnalogParams) -> int:
        max_lag = params.horizon + params.shape_days + params.archive_days - 1
        feature_history = max(200, params.volatility_days + 1,
                              2 * params.roc_days + params.shape_days,
                              params.atr_days + 1)
        return int(max_lag + feature_history)

    @staticmethod
    def indicators(data: pd.DataFrame, params: AnalogParams) -> pd.DataFrame:
        if min(params.roc_days, params.shape_days, params.archive_days,
               params.neighbors, params.horizon, params.volatility_days,
               params.atr_days) < 1:
            raise ValueError('Lookbacks and neighbor count must be positive')
        if params.neighbors > params.archive_days:
            raise ValueError('Neighbor count exceeds archive length')
        if not (np.isfinite(params.stop_atr) and params.stop_atr > 0
                and np.isfinite(params.max_distance) and params.max_distance > 0
                and np.isfinite(params.forecast_threshold)
                and params.forecast_threshold > 0):
            raise ValueError('Thresholds and stop multiplier must be finite and positive')
        out = pd.DataFrame(index=data.index)
        close = data['close']
        out['daily_return'] = close.pct_change(fill_method=None)
        out['volatility'] = out['daily_return'].rolling(
            params.volatility_days, min_periods=params.volatility_days
        ).std(ddof=0)
        out['roc'] = close.pct_change(params.roc_days, fill_method=None)
        out['acceleration'] = out['roc'].diff(params.roc_days)
        denominator = out['volatility'].replace(0.0, np.nan) * np.sqrt(2 * params.roc_days)
        out['normalized_acceleration'] = out['acceleration'] / denominator
        out['ma200'] = close.rolling(200, min_periods=200).mean()
        out['regime'] = np.where(out['ma200'].notna(),
                                 np.sign(close - out['ma200']), np.nan)
        previous_close = close.shift(1)
        out['true_range'] = pd.concat([
            data['high'] - data['low'],
            (data['high'] - previous_close).abs(),
            (data['low'] - previous_close).abs()
        ], axis=1).max(axis=1)
        out['atr'] = out['true_range'].rolling(
            params.atr_days, min_periods=params.atr_days
        ).mean()
        out['realized_return'] = close.pct_change(params.horizon, fill_method=None)
        for offset in range(params.shape_days):
            out[f'shape_{offset}'] = out['normalized_acceleration'].shift(offset)

        n = len(data)
        distances = np.full((n, params.archive_days), np.inf)
        outcomes = np.full((n, params.archive_days), np.nan)
        current_regime = out['regime'].to_numpy()
        # ponytail: O(bars * archive * shape); chunk the archive if memory becomes limiting.
        for column in range(params.archive_days):
            lag = params.horizon + params.shape_days + column
            squared_distance = np.zeros(n)
            for offset in range(params.shape_days):
                feature = out[f'shape_{offset}']
                difference = (feature - feature.shift(lag)).to_numpy()
                squared_distance += difference * difference
            distance = np.sqrt(squared_distance / params.shape_days)
            historical_regime = out['regime'].shift(lag).to_numpy()
            # This endpoint's forward outcome ended at least shape_days bars ago.
            outcome = out['realized_return'].shift(lag - params.horizon).to_numpy()
            valid = (np.isfinite(distance) & np.isfinite(outcome)
                     & np.isfinite(current_regime)
                     & (current_regime != 0)
                     & (current_regime == historical_regime))
            distances[:, column] = np.where(valid, distance, np.inf)
            outcomes[:, column] = outcome

        nearest = np.argpartition(distances, params.neighbors - 1, axis=1)[:, :params.neighbors]
        selected_distances = np.take_along_axis(distances, nearest, axis=1)
        selected_outcomes = np.take_along_axis(outcomes, nearest, axis=1)
        complete = np.isfinite(selected_distances).all(axis=1)
        out['analog_count'] = np.isfinite(selected_distances).sum(axis=1)
        out['analog_distance'] = np.where(
            complete, selected_distances.max(axis=1), np.nan
        )
        out['forecast'] = np.where(
            complete, selected_outcomes.mean(axis=1), np.nan
        )
        return out.replace([np.inf, -np.inf], np.nan)

    @staticmethod
    def generate_signals(data: pd.DataFrame, indicators: pd.DataFrame,
                         ctx: StrategyContext, params: AnalogParams) -> SignalFrame:
        close = data['close'].to_numpy(dtype=float)
        opens = data['open'].to_numpy(dtype=float)
        atr = indicators['atr'].to_numpy(dtype=float)
        forecast = indicators['forecast'].to_numpy(dtype=float)
        distance = indicators['analog_distance'].to_numpy(dtype=float)
        regime = indicators['regime'].to_numpy(dtype=float)
        ready = (np.arange(len(data)) >= GeneratedStrategy.warmup_bars(params))
        eligible = (ready & np.isfinite(close) & np.isfinite(atr) & (atr > 0)
                    & np.isfinite(forecast) & np.isfinite(distance)
                    & (distance <= params.max_distance))
        candidates = np.zeros(len(data), dtype=int)
        candidates[eligible & (regime == 1)
                   & (forecast >= params.forecast_threshold)] = 1
        candidates[eligible & (regime == -1)
                   & (forecast <= -params.forecast_threshold)] = -1

        decisions = np.zeros(len(data), dtype=int)
        position = 0
        pending = 0
        entry_price = 0.0
        entry_atr = 0.0
        entry_bar = 0
        for i in range(len(data)):
            # Apply yesterday's decision at today's open, before inspecting today's close.
            if pending != position:
                position = pending
                if position:
                    if i == 0 or not np.isfinite(opens[i]) or opens[i] <= 0:
                        position = 0
                        pending = 0
                        continue
                    entry_price = opens[i]
                    entry_atr = atr[i - 1]
                    entry_bar = i
            if position:
                stopped = (np.isfinite(close[i]) and
                           position * (close[i] - entry_price) < -params.stop_atr * entry_atr)
                expired = i - entry_bar + 1 >= params.horizon
                decision = 0 if stopped or expired else position
            else:
                decision = int(candidates[i])
            decisions[i] = decision
            pending = decision

        df = pd.DataFrame({'signal': decisions, 'size': 1.0}, index=data.index)
        df['signal'] = df['signal'].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column='signal', size_column='size')


def _self_check():
    params = AnalogParams()
    start = GeneratedStrategy.warmup_bars(params)
    index = pd.date_range('2020-01-01', periods=start + 6, freq='B')
    for direction in (1, -1):
        data = pd.DataFrame({'open': 100.0, 'high': 101.0, 'low': 99.0,
                             'close': 100.0, 'volume': 1000.0}, index=index)
        features = pd.DataFrame({'atr': 1.0, 'forecast': 0.0,
                                 'analog_distance': 0.1, 'regime': float(direction)}, index=index)
        features.iloc[start, features.columns.get_loc('forecast')] = direction * 0.02
        # Entry open differs from the decision close; the stop must use the entry open.
        data.iloc[start + 1, data.columns.get_loc('open')] = 100.0 + direction * 5.0
        data.iloc[start + 1, data.columns.get_loc('close')] = 100.0 + direction * 5.0
        data.iloc[start + 2, data.columns.get_loc('close')] = 100.0 + direction * 2.0
        result = GeneratedStrategy.generate_signals(data, features, None, params).data
        assert result['signal'].iloc[start] == 0
        assert result['signal'].iloc[start + 1] == direction
        assert result['signal'].iloc[start + 2] == direction
        assert result['signal'].iloc[start + 3] == 0
        assert (result['size'] > 0).all()


if __name__ == '__main__':
    _self_check()
