from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class AnalogParams:
    trend_bars: int = 18
    shape_spacing: int = 6
    volatility_bars: int = 20
    history_bars: int = 252
    neighbors: int = 12
    forecast_threshold: float = 0.012
    innovation_threshold: float = 0.15
    hold_bars: int = 18
    profit_target: float = 0.045


class GeneratedStrategy(BaseStrategy[AnalogParams]):
    strategy_id = 'gen_local-w2_1791222827'

    @classmethod
    def params_type(cls) -> type[AnalogParams]:
        return AnalogParams

    def warmup_bars(self, params: AnalogParams) -> int:
        return int(max(params.trend_bars, params.volatility_bars) + 1
                   + 2 * params.shape_spacing
                   + params.hold_bars + params.history_bars)

    def indicators(self, data: pd.DataFrame, params: AnalogParams) -> pd.DataFrame:
        if (min(params.trend_bars, params.shape_spacing, params.history_bars,
                params.neighbors, params.hold_bars) < 1
                or params.volatility_bars < 2
                or params.neighbors > params.history_bars
                or params.forecast_threshold <= 0
                or params.innovation_threshold <= 0
                or params.profit_target <= 0):
            raise ValueError('Invalid strategy parameters')
        result = pd.DataFrame(index=data.index)
        result['log_close'] = np.log(data['close'].where(data['close'] > 0))
        result['log_return'] = result['log_close'].diff()
        result['volatility'] = result['log_return'].rolling(
            params.volatility_bars).std().replace(0.0, np.nan)
        result['trend_slope'] = result['log_return'].rolling(
            params.trend_bars).mean()
        result['normalized_slope'] = result['trend_slope'] / result['volatility']
        result['slope_middle'] = result['normalized_slope'].shift(params.shape_spacing)
        result['slope_oldest'] = result['normalized_slope'].shift(2 * params.shape_spacing)
        result['slope_innovation'] = (
            result['trend_slope'] - result['trend_slope'].shift(params.shape_spacing)
        ) / result['volatility']
        result['completed_return'] = result['log_close'].diff(params.hold_bars)

        features = result[['normalized_slope', 'slope_middle', 'slope_oldest']]
        distances = np.full((len(data), params.history_bars), np.inf)
        outcomes = np.zeros_like(distances)
        for column, lag in enumerate(range(params.hold_bars + 1,
                                           params.hold_bars + params.history_bars + 1)):
            historical = features.shift(lag)
            distance = ((features - historical) ** 2).sum(axis=1, skipna=False)
            outcome = result['completed_return'].shift(lag - params.hold_bars)
            valid = np.isfinite(distance.to_numpy()) & np.isfinite(outcome.to_numpy())
            distances[valid, column] = distance.to_numpy()[valid]
            outcomes[valid, column] = outcome.to_numpy()[valid]

        chosen = np.argpartition(distances, params.neighbors - 1, axis=1)[:, :params.neighbors]
        selected_distances = np.take_along_axis(distances, chosen, axis=1)
        selected_outcomes = np.take_along_axis(outcomes, chosen, axis=1)
        weights = 1.0 / (1.0 + selected_distances)
        total_weight = weights.sum(axis=1)
        forecast = np.divide((weights * selected_outcomes).sum(axis=1), total_weight,
                             out=np.full(len(data), np.nan), where=total_weight > 0)
        forecast[np.isfinite(selected_distances).sum(axis=1) < params.neighbors] = np.nan
        result['analog_forecast'] = forecast
        return result.replace([np.inf, -np.inf], np.nan)

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame,
                         ctx: StrategyContext, params: AnalogParams) -> SignalFrame:
        forecast = indicators['analog_forecast']
        innovation = indicators['slope_innovation']
        long_entry = ((forecast > params.forecast_threshold)
                      & (innovation > params.innovation_threshold))
        short_entry = ((forecast < -params.forecast_threshold)
                       & (innovation < -params.innovation_threshold))
        entries = np.select([long_entry, short_entry], [1, -1], default=0)
        decisions = np.zeros(len(data), dtype=int)
        opens = data['open'].to_numpy(dtype=float)
        closes = data['close'].to_numpy(dtype=float)
        position = 0
        entry_bar = -1
        entry_price = np.nan
        warmup = self.warmup_bars(params)

        for i in range(len(data)):
            if position:
                if i == entry_bar:
                    entry_price = opens[i]
                if not np.isfinite(entry_price) or entry_price <= 0:
                    position = 0
                    continue
                gain = position * (closes[i] / entry_price - 1.0)
                if gain >= params.profit_target or i - entry_bar + 1 >= params.hold_bars:
                    position = 0
                    continue
                decisions[i] = position
            elif i >= warmup - 1 and i + 1 < len(data) and entries[i]:
                position = int(entries[i])
                entry_bar = i + 1
                entry_price = np.nan
                decisions[i] = position

        df = pd.DataFrame({'signal': decisions, 'size': 0.95}, index=data.index)
        df['signal'] = df['signal'].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column='signal', size_column='size')


def _self_check() -> None:
    params = AnalogParams()
    n = 450
    index = pd.date_range('2020-01-01', periods=n, freq='B')
    data = pd.DataFrame({'open': 100.0, 'high': 106.0, 'low': 99.0,
                         'close': 100.0, 'volume': 1000000.0}, index=index)
    strategy = GeneratedStrategy()
    derived = strategy.indicators(data, params)
    assert derived.index.equals(data.index)
    forced = pd.DataFrame({'analog_forecast': 0.03, 'slope_innovation': 0.3}, index=index)
    first_fill = strategy.warmup_bars(params)
    data.iloc[first_fill, data.columns.get_loc('close')] = 105.0
    signals = strategy.generate_signals(data, forced, None, params).data
    assert signals['signal'].iloc[:first_fill].eq(0).all()
    assert signals['signal'].iloc[first_fill] == 1
    assert signals['signal'].iloc[first_fill + 1] == 0
    data['close'] = 100.0
    signals = strategy.generate_signals(data, forced, None, params).data
    assert signals['signal'].iloc[first_fill:first_fill + params.hold_bars].eq(1).all()
    assert signals['signal'].iloc[first_fill + params.hold_bars] == 0
    forced['analog_forecast'] = -0.03
    forced['slope_innovation'] = -0.3
    data.iloc[first_fill, data.columns.get_loc('close')] = 95.0
    signals = strategy.generate_signals(data, forced, None, params).data
    assert signals['signal'].iloc[first_fill] == -1
    assert signals['signal'].iloc[first_fill + 1] == 0
    forced['slope_innovation'] = 0.3
    assert strategy.generate_signals(data, forced, None, params).data['signal'].eq(0).all()
    assert signals['signal'].isin([-1, 0, 1]).all() and signals['size'].gt(0).all()


if __name__ == '__main__':
    _self_check()
