from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class PropagationParams:
    lag: int = 5
    memory: int = 42
    rank_window: int = 126
    entry_percentile: float = 0.85
    persistence_percentile: float = 0.60
    atr_window: int = 14
    trail_atr: float = 3.0
    holding_bars: int = 18

    def __post_init__(self):
        for value in (self.lag, self.memory, self.rank_window, self.atr_window, self.holding_bars):
            if not isinstance(value, int) or isinstance(value, bool) or value < 1:
                raise ValueError('Lookbacks and holding_bars must be positive integers.')
        if self.memory < 3 or self.rank_window < 3:
            raise ValueError('Memory and rank windows must be at least three.')
        if not 0.5 < self.entry_percentile < 1.0:
            raise ValueError('entry_percentile must be between 0.5 and 1.')
        if not 0.0 < self.persistence_percentile < 1.0:
            raise ValueError('persistence_percentile must be between 0 and 1.')
        if not np.isfinite(self.trail_atr) or self.trail_atr <= 0:
            raise ValueError('trail_atr must be finite and positive.')


class GeneratedStrategy(BaseStrategy[PropagationParams]):
    strategy_id = 'gen_local-w2_1791215178'

    @classmethod
    def params_type(cls):
        return PropagationParams

    def warmup_bars(self, params):
        return int(max(params.lag + params.memory + params.rank_window + 1, params.atr_window + 1))

    def indicators(self, data, params):
        out = pd.DataFrame(index=data.index)
        close = data['close']
        out['change'] = close.diff()
        out['sign'] = np.sign(out['change'])
        out['lag_sign'] = out['sign'].shift(params.lag)
        valid = out['sign'].notna() & out['lag_sign'].notna()
        out['up_parent'] = (out['lag_sign'] > 0).astype(float).where(valid)
        out['down_parent'] = (out['lag_sign'] < 0).astype(float).where(valid)
        out['up_survival'] = ((out['lag_sign'] > 0) & (out['sign'] > 0)).astype(float).where(valid)
        out['down_survival'] = ((out['lag_sign'] < 0) & (out['sign'] < 0)).astype(float).where(valid)
        for side in ('up', 'down'):
            out[side + '_parents'] = out[side + '_parent'].rolling(params.memory).sum()
            out[side + '_survivors'] = out[side + '_survival'].rolling(params.memory).sum()
            out[side + '_probability'] = out[side + '_survivors'] / out[side + '_parents'].replace(0.0, np.nan)
        out['imbalance'] = out['up_probability'] - out['down_probability']
        out['imbalance_rank'] = out['imbalance'].rolling(params.rank_window).rank(pct=True)
        out['persistence'] = out['sign'].rolling(params.memory).corr(out['lag_sign'])
        out['persistence_rank'] = out['persistence'].rolling(params.rank_window).rank(pct=True)
        out['previous_close'] = close.shift(1)
        out['true_range'] = pd.concat([
            data['high'] - data['low'],
            (data['high'] - out['previous_close']).abs(),
            (data['low'] - out['previous_close']).abs(),
        ], axis=1).max(axis=1)
        out['atr'] = out['true_range'].rolling(params.atr_window).mean()
        ready = np.isfinite(out[['imbalance_rank', 'persistence_rank', 'atr']]).all(axis=1) & (out['atr'] > 0)
        active = ready & (out['persistence_rank'] >= params.persistence_percentile)
        out['entry'] = np.select([
            active & (out['imbalance_rank'] >= params.entry_percentile),
            active & (out['imbalance_rank'] <= 1.0 - params.entry_percentile),
        ], [1, -1], default=0).astype(int)
        return out.replace([np.inf, -np.inf], np.nan)

    def generate_signals(self, data, indicators, ctx, params):
        df = pd.DataFrame(index=data.index)
        signals = np.zeros(len(data), dtype=int)
        position, age = 0, 0
        high, low = np.nan, np.nan
        high_stop, short_stop = -np.inf, np.inf
        closes = data['close'].to_numpy(dtype=float)
        opens = data['open'].to_numpy(dtype=float)
        atrs = indicators['atr'].to_numpy(dtype=float)
        entries = indicators['entry'].fillna(0).to_numpy(dtype=int)
        for i in range(len(data)):
            price, atr = closes[i], atrs[i]
            usable = np.isfinite(price) and np.isfinite(atr) and atr > 0
            if position:
                age += 1
                exit_now = not usable
                if usable:
                    if np.isnan(high):
                        # The prior decision fills at today's open.
                        entry_price = opens[i] if np.isfinite(opens[i]) else price
                        high, low = max(entry_price, price), min(entry_price, price)
                    else:
                        high, low = max(high, price), min(low, price)
                    high_stop = max(high_stop, high - params.trail_atr * atr)
                    short_stop = min(short_stop, low + params.trail_atr * atr)
                    # Required close high-water exit applies to both directions.
                    # Shorts also have a mirrored adverse-price ratchet.
                    exit_now = price <= high_stop or (position < 0 and price >= short_stop)
                if exit_now or age >= params.holding_bars:
                    position = 0
            elif usable and entries[i] in (-1, 1):
                position, age = int(entries[i]), 0
                high, low = np.nan, np.nan
                high_stop, short_stop = -np.inf, np.inf
            signals[i] = position
        df['signal'] = signals
        df['signal'] = df['signal'].shift(1).fillna(0).astype(int)
        df['size'] = 0.95
        return SignalFrame(data=df, signal_column='signal', size_column='size')


def _self_check():
    strategy = GeneratedStrategy()
    params = PropagationParams()
    index = pd.date_range('2020-01-01', periods=7, freq='B')
    close = np.array([100., 100., 110., 108., 105., 100., 100.])
    data = pd.DataFrame({'open': close, 'high': close + 1., 'low': close - 1., 'close': close, 'volume': 1000.}, index=index)
    ind = pd.DataFrame({'atr': [1., 1., 1., 5., 5., 1., 1.], 'entry': [1, 0, 0, 0, 0, 0, 0]}, index=index)
    result = strategy.generate_signals(data, ind, None, params).data
    assert result['signal'].tolist() == [0, 1, 1, 1, 1, 0, 0]
    assert (result['size'] > 0).all()
    params.holding_bars = 3
    ind['atr'] = 100.
    assert strategy.generate_signals(data, ind, None, params).data['signal'].tolist() == [0, 1, 1, 1, 0, 0, 0]
    params.holding_bars = 18
    data['close'] = [100., 100., 102., 104., 104., 104., 104.]
    data['open'] = data['close']
    ind['atr'] = 1.
    ind['entry'] = [-1, 0, 0, 0, 0, 0, 0]
    assert strategy.generate_signals(data, ind, None, params).data['signal'].tolist() == [0, -1, -1, -1, 0, 0, 0]
    derived = strategy.indicators(data, params)
    assert derived.index.equals(data.index) and (derived['entry'] == 0).all()
    assert strategy.warmup_bars(params) >= params.lag + params.memory + params.rank_window


if __name__ == '__main__':
    _self_check()
