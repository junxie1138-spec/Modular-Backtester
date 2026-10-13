from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class ErasureParams:
    gain_days: int = 3
    volatility_days: int = 20
    atr_days: int = 14
    erasure_threshold: float = 1.25
    shock_threshold: float = 1.2
    breakeven_pct: float = 0.005
    trail_atr: float = 0.75
    max_hold: int = 2
    minimum_size: float = 0.25
    maximum_size: float = 0.95


class GeneratedStrategy(BaseStrategy[ErasureParams]):
    strategy_id = 'gen_local-w2_1790970718'

    @classmethod
    def params_type(cls):
        return ErasureParams

    def warmup_bars(self, params):
        return int(max(params.gain_days + 2, params.volatility_days + 2, params.atr_days + 1))

    def indicators(self, data, params):
        if min(params.gain_days, params.volatility_days, params.atr_days) < 2:
            raise ValueError('Lookbacks must be at least two bars')
        if params.max_hold not in (1, 2):
            raise ValueError('max_hold must be one or two sessions')
        if not (0 < params.minimum_size <= params.maximum_size <= 1):
            raise ValueError('Sizes must satisfy 0 < minimum <= maximum <= 1')
        if not all(np.isfinite(x) and x > 0 for x in (params.erasure_threshold, params.shock_threshold, params.breakeven_pct, params.trail_atr)):
            raise ValueError('Thresholds must be finite and positive')
        close = data['close'].where(data['close'] > 0)
        out = pd.DataFrame(index=data.index)
        out['return'] = close.pct_change(fill_method=None)
        out['log_return'] = np.log(close).diff()
        out['prior_gain'] = np.expm1(out['log_return'].shift(1).rolling(params.gain_days).sum())
        out['prior_min_return'] = out['return'].shift(1).rolling(params.gain_days).min()
        out['volatility'] = out['return'].shift(1).rolling(params.volatility_days).std(ddof=0)
        out['true_range'] = pd.concat([(data['high'] - data['low']), (data['high'] - close.shift(1)).abs(), (data['low'] - close.shift(1)).abs()], axis=1).max(axis=1)
        out['atr'] = out['true_range'].rolling(params.atr_days).mean()
        out['erasure'] = -out['return'] / out['prior_gain'].where(out['prior_gain'] > 0)
        out['shock'] = -out['return'] / out['volatility'].where(out['volatility'] > 0)
        out = out.replace([np.inf, -np.inf], np.nan)
        out['strength'] = (out['erasure'] / params.erasure_threshold - 1.0).clip(0.0, 1.0).fillna(0.0)
        out['candidate'] = ((out['prior_min_return'] > 0) & (out['erasure'] >= params.erasure_threshold) & (out['shock'] >= params.shock_threshold) & (out['atr'] > 0)).fillna(False)
        return out

    def generate_signals(self, data, indicators, ctx, params):
        count = len(data)
        signals = np.zeros(count, dtype=int)
        sizes = np.full(count, params.minimum_size, dtype=float)
        active = False
        entry = np.nan
        stop = -np.inf
        peak = -np.inf
        armed = False
        age = 0
        position_size = params.minimum_size
        for i in range(count):
            opening = float(data['open'].iloc[i])
            high = float(data['high'].iloc[i])
            close = float(data['close'].iloc[i])
            atr = float(indicators['atr'].iloc[i])
            valid = all(np.isfinite(x) and x > 0 for x in (opening, high, close, atr))
            if active:
                if np.isnan(entry):
                    # The preceding close's decision fills at this session's open.
                    entry = opening
                    peak = entry
                age += 1
                if valid:
                    peak = max(peak, high)
                    armed = armed or peak >= entry * (1.0 + params.breakeven_pct)
                    if armed:
                        stop = max(stop, entry, peak - params.trail_atr * atr)
                # Stops are evaluated at the close and executed next open.
                # ponytail: opening price excludes fill costs; use recorded fills if available.
                if not valid or (armed and close <= stop) or age >= params.max_hold:
                    active = False
                else:
                    signals[i] = 1
                    sizes[i] = position_size
            elif valid and bool(indicators['candidate'].iloc[i]):
                strength = float(indicators['strength'].iloc[i])
                if not np.isfinite(strength):
                    continue
                position_size = params.minimum_size + (params.maximum_size - params.minimum_size) * float(np.clip(strength, 0.0, 1.0))
                active = True
                entry = np.nan
                stop = -np.inf
                peak = -np.inf
                armed = False
                age = 0
                signals[i] = 1
                sizes[i] = position_size
        df = pd.DataFrame({'signal': signals, 'size': sizes}, index=data.index)
        df['signal'] = df['signal'].shift(1).fillna(0).astype(int)
        df['size'] = df['size'].shift(1).fillna(params.minimum_size).astype(float)
        return SignalFrame(data=df, signal_column='signal', size_column='size')


def _self_check():
    index = pd.date_range('2020-01-01', periods=5)
    data = pd.DataFrame({'open': [100.0] * 5, 'high': [100.2] * 5, 'low': [99.0] * 5, 'close': [100.0] * 5, 'volume': [1000.0] * 5}, index=index)
    derived = pd.DataFrame({'atr': [1.0] * 5, 'candidate': [True, False, False, False, False], 'strength': [0.5] * 5}, index=index)
    params = ErasureParams()
    strategy = GeneratedStrategy()
    result = strategy.generate_signals(data, derived, None, params).data
    assert result['signal'].tolist() == [0, 1, 1, 0, 0]
    assert np.isclose(result['size'].iloc[1], 0.6)
    data.loc[index[1], 'high'] = 102.0
    data.loc[index[1], 'close'] = 101.0
    result = strategy.generate_signals(data, derived, None, params).data
    assert result['signal'].tolist() == [0, 1, 0, 0, 0]
    assert (result['size'] > 0).all()


if __name__ == '__main__':
    _self_check()
