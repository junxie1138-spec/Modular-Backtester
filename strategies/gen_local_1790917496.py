from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class OrdinalParams:
    pattern_window: int = 60
    correlation_window: int = 30
    max_complexity: float = 0.85
    min_pattern_count: int = 6
    atr_window: int = 14
    stop_atr: float = 1.5
    hold_days: int = 4

    def __post_init__(self):
        if min(self.pattern_window, self.correlation_window, self.atr_window) < 2:
            raise ValueError('Lookbacks must be at least two.')
        if not 0 < self.max_complexity <= 1:
            raise ValueError('Complexity must be in (0, 1].')
        if not 1 <= self.min_pattern_count <= self.pattern_window:
            raise ValueError('Invalid minimum pattern count.')
        if not np.isfinite(self.stop_atr) or self.stop_atr <= 0:
            raise ValueError('Stop distance must be positive and finite.')
        if not 3 <= self.hold_days <= 5:
            raise ValueError('Holding horizon must be three to five sessions.')


class GeneratedStrategy(BaseStrategy[OrdinalParams]):
    strategy_id = 'gen_local_1790917496'

    @classmethod
    def params_type(cls):
        return OrdinalParams

    def warmup_bars(self, params: OrdinalParams) -> int:
        return int(max(params.pattern_window + 4,
                       params.correlation_window + 2,
                       params.atr_window + 1))

    def indicators(self, data: pd.DataFrame, params: OrdinalParams) -> pd.DataFrame:
        out = pd.DataFrame(index=data.index)
        r = data['close'].pct_change(fill_method=None).replace([np.inf, -np.inf], np.nan)
        x, y, z = r.shift(2), r.shift(1), r
        valid = x.notna() & y.notna() & z.notna()
        pattern = pd.Series(np.select([
            (x <= y) & (y <= z),
            (x <= z) & (z < y),
            (y < x) & (x <= z),
            (y <= z) & (z < x),
            (z < x) & (x <= y),
            (z < y) & (y < x),
        ], [0, 1, 2, 3, 4, 5], default=np.nan), index=data.index).where(valid)
        out['return'] = r
        out['pattern'] = pattern
        collision = pd.Series(0.0, index=data.index)
        forecast = pd.Series(np.nan, index=data.index)
        for code in range(6):
            member = pattern.eq(code).astype(float).where(valid)
            probability = member.rolling(params.pattern_window).mean()
            previous_member = member.shift(1)
            count = previous_member.rolling(params.pattern_window).sum()
            total = (previous_member * r).rolling(params.pattern_window).sum()
            conditional = (total / count.replace(0, np.nan)).where(count >= params.min_pattern_count)
            out[f'pattern_probability_{code}'] = probability
            out[f'pattern_count_{code}'] = count
            out[f'pattern_forecast_{code}'] = conditional
            collision = collision + probability.pow(2)
            forecast = forecast.where(pattern.ne(code), conditional)
        out['complexity'] = 1.0 / (6.0 * collision.replace(0, np.nan))
        out['ordinal_forecast'] = forecast
        out['autocorrelation'] = r.rolling(params.correlation_window).corr(r.shift(1))
        out['correlation_forecast'] = out['autocorrelation'] * r
        previous_close = data['close'].shift(1)
        out['true_range'] = pd.concat([
            data['high'] - data['low'],
            (data['high'] - previous_close).abs(),
            (data['low'] - previous_close).abs(),
        ], axis=1).max(axis=1)
        out['atr'] = out['true_range'].rolling(params.atr_window).mean()
        return out.replace([np.inf, -np.inf], np.nan)

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame,
                         ctx: StrategyContext, params: OrdinalParams) -> SignalFrame:
        ordinal = indicators['ordinal_forecast']
        correlation = indicators['correlation_forecast']
        eligible = (indicators['complexity'] <= params.max_complexity) & (indicators['atr'] > 0)
        candidate = np.select([
            eligible & (ordinal > 0) & (correlation > 0),
            eligible & (ordinal < 0) & (correlation < 0),
        ], [1, -1], default=0)
        closes = data['close'].to_numpy(dtype=float)
        opens = data['open'].to_numpy(dtype=float)
        atr = indicators['atr'].to_numpy(dtype=float)
        raw = np.zeros(len(data), dtype=int)
        position = 0
        pending = False
        age = 0
        entry = 0.0
        distance = 0.0
        for i in range(len(data)):
            if position:
                if pending:
                    # The prior close's decision fills at this session's open.
                    entry = opens[i]
                    distance = params.stop_atr * atr[i - 1]
                    age = 0
                    pending = False
                age += 1
                invalid = not (np.isfinite(entry) and np.isfinite(distance) and np.isfinite(closes[i]))
                stopped = position * (closes[i] - entry) <= -distance
                if invalid or stopped or age >= params.hold_days:
                    position = 0
                # A stopped or expired trade cannot re-enter on this close.
            elif candidate[i] and np.isfinite(closes[i]) and np.isfinite(atr[i]):
                position = int(candidate[i])
                pending = True
            raw[i] = position
        df = pd.DataFrame({'signal': raw, 'size': 1.0}, index=data.index)
        df['signal'] = df['signal'].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column='signal', size_column='size')


if __name__ == '__main__':
    index = pd.date_range('2020-01-01', periods=10)
    data = pd.DataFrame({'open': 100.0, 'high': 104.0, 'low': 96.0,
                         'close': [100., 100., 97., 100., 100., 103., 100., 100., 100., 100.],
                         'volume': 1000.0}, index=index)
    ind = pd.DataFrame({'ordinal_forecast': [1., 0., 0., -1., 0., 0., 0., 0., 0., 0.],
                        'correlation_forecast': [1., 0., 0., -1., 0., 0., 0., 0., 0., 0.],
                        'complexity': 0.5, 'atr': 1.0}, index=index)
    strategy = GeneratedStrategy()
    result = strategy.generate_signals(data, ind, None, OrdinalParams()).data
    assert result['signal'].tolist() == [0, 1, 1, 0, -1, -1, 0, 0, 0, 0]
    flat = data.copy()
    flat['close'] = 100.0
    ind['ordinal_forecast'] = [1.] + [0.] * 9
    ind['correlation_forecast'] = ind['ordinal_forecast']
    result = strategy.generate_signals(flat, ind, None, OrdinalParams()).data
    assert result['signal'].tolist() == [0, 1, 1, 1, 1, 0, 0, 0, 0, 0]
    ind['correlation_forecast'] = -ind['ordinal_forecast']
    assert strategy.generate_signals(flat, ind, None, OrdinalParams()).data['signal'].eq(0).all()
