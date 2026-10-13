from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class ShapeParams:
    path_bars: int = 8
    history_bars: int = 252
    neighbors: int = 15
    forecast_threshold: float = 0.002
    profit_target: float = 0.015
    hold_bars: int = 4


class GeneratedStrategy(BaseStrategy[ShapeParams]):
    strategy_id = 'gen_local_1790971199'

    @classmethod
    def params_type(cls):
        return ShapeParams

    @staticmethod
    def warmup_bars(params):
        return int(params.history_bars + 2 * params.path_bars + params.hold_bars + 1)

    def indicators(self, data, params):
        p = params
        if not (p.path_bars >= 2 and p.history_bars >= p.neighbors >= 2):
            raise ValueError('Invalid path length or analog sample sizes')
        if not (3 <= p.hold_bars <= 5 and p.forecast_threshold > 0 and p.profit_target > 0):
            raise ValueError('Invalid holding horizon or return thresholds')
        close = data['close'].astype(float).where(data['close'] > 0)
        opening = data['open'].astype(float).where(data['open'] > 0)
        returns = np.log(close).diff().replace([np.inf, -np.inf], np.nan)
        variation = returns.abs().rolling(p.path_bars, min_periods=p.path_bars).sum()
        denominator = variation.where(variation > 0)
        displacement = returns.rolling(p.path_bars, min_periods=p.path_bars).sum()
        # Antisymmetric second-order signature of (elapsed time, log price).
        # Oldest return receives weight 1-L; newest receives weight L-1.
        area_numerator = pd.Series(0.0, index=data.index)
        for lag in range(p.path_bars):
            area_numerator = area_numerator + (p.path_bars - 1 - 2 * lag) * returns.shift(lag)
        area = area_numerator / (2.0 * p.path_bars * denominator)
        efficiency = displacement / denominator
        out = pd.DataFrame(index=data.index)
        out['log_return'] = returns
        out['variation'] = variation
        out['displacement'] = displacement
        out['area_numerator'] = area_numerator
        out['area'] = area
        out['efficiency'] = efficiency
        # At maturity m, this outcome belongs to the shape ending at m-h-1.
        # Entry and expiry are both next-open fills, exactly h sessions apart.
        out['matured_return'] = opening / opening.shift(p.hold_bars) - 1.0
        out['matured_area'] = area.shift(p.hold_bars + 1)
        out['matured_efficiency'] = efficiency.shift(p.hold_bars + 1)
        out = out.replace([np.inf, -np.inf], np.nan)
        current = out[['area', 'efficiency']].to_numpy(dtype=float)
        historical = out[['matured_area', 'matured_efficiency']].to_numpy(dtype=float)
        outcomes = out['matured_return'].to_numpy(dtype=float)
        forecast = np.full(len(data), np.nan)
        # ponytail: O(bars * history) exact search; use an index only for much larger histories.
        for i in range(self.warmup_bars(p) - 1, len(data)):
            if not np.isfinite(current[i]).all():
                continue
            # Embargo all analog outcomes from the current shape's return window.
            end = i - p.path_bars + 1
            start = max(0, end - p.history_bars)
            sample = historical[start:end]
            target = outcomes[start:end]
            valid = np.isfinite(sample).all(axis=1) & np.isfinite(target)
            if valid.sum() < p.neighbors:
                continue
            delta = sample[valid] - current[i]
            distance = (2.0 * delta[:, 0]) ** 2 + (0.5 * delta[:, 1]) ** 2
            selected = np.argsort(distance, kind='stable')[:p.neighbors]
            forecast[i] = float(target[valid][selected].mean())
        out['forecast'] = forecast
        return out

    def generate_signals(self, data, indicators, ctx, params):
        p = params
        forecasts = indicators['forecast'].to_numpy(dtype=float)
        direction = np.zeros(len(data), dtype=int)
        valid = np.isfinite(forecasts)
        direction[valid & (forecasts >= p.forecast_threshold)] = 1
        direction[valid & (forecasts <= -p.forecast_threshold)] = -1
        opening = data['open'].to_numpy(dtype=float)
        closing = data['close'].to_numpy(dtype=float)
        decisions = np.zeros(len(data), dtype=int)
        position = 0
        entry_bar = -1
        entry_price = np.nan
        for i in range(len(data)):
            if position != 0:
                if i == entry_bar:
                    entry_price = opening[i]
                if not np.isfinite(entry_price) or entry_price <= 0:
                    position = 0
                else:
                    gain = position * (closing[i] / entry_price - 1.0)
                    target_hit = np.isfinite(gain) and gain >= p.profit_target
                    expired = i - entry_bar + 1 >= p.hold_bars
                    if target_hit or expired:
                        position = 0
            elif i > 0 and direction[i] != 0 and direction[i] == direction[i - 1]:
                if np.isfinite(closing[i]) and closing[i] > 0:
                    position = int(direction[i])
                    entry_bar = i + 1
                    entry_price = np.nan
            decisions[i] = position
        df = pd.DataFrame({'signal': decisions, 'size': 1.0}, index=data.index)
        # Targets are observed at the close; exits execute at the following open.
        df['signal'] = df['signal'].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column='signal', size_column='size')


def _self_check():
    strategy = GeneratedStrategy()
    params = ShapeParams()
    index = pd.date_range('2020-01-01', periods=10)
    data = pd.DataFrame({'open': 100.0, 'high': 103.0, 'low': 97.0, 'close': 100.0, 'volume': 1000.0}, index=index)
    for side in (1, -1):
        indicators = pd.DataFrame({'forecast': side * 0.01}, index=index)
        result = strategy.generate_signals(data, indicators, None, params).data
        expected = np.array([0, 0, 1, 1, 1, 1, 0, 0, 1, 1]) * side
        assert np.array_equal(result['signal'].to_numpy(), expected)
        hit = data.copy()
        hit.loc[index[2], 'close'] = 100.0 * (1.0 + side * 0.02)
        result = strategy.generate_signals(hit, indicators, None, params).data
        assert result['signal'].iloc[2] == side
        assert result['signal'].iloc[3] == 0
        assert (result['size'] > 0).all()
    params = ShapeParams(path_bars=4, history_bars=20, neighbors=5)
    index = pd.date_range('2020-01-01', periods=90)
    price = 100.0 * np.exp(np.cumsum(0.006 * np.sin(np.arange(90) * 0.7)))
    data = pd.DataFrame({'open': price * 0.999, 'high': price * 1.01, 'low': price * 0.99, 'close': price, 'volume': 1000.0}, index=index)
    full = strategy.indicators(data, params)
    prefix = strategy.indicators(data.iloc[:70], params)
    assert np.isfinite(full['forecast'].iloc[-1])
    assert np.allclose(full.iloc[:70].to_numpy(), prefix.to_numpy(), equal_nan=True)


if __name__ == '__main__':
    _self_check()
