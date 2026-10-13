from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class WaveletParams:
    coherence_floor: float = 0.0
    drift_floor: float = 0.0001


class GeneratedStrategy(BaseStrategy[WaveletParams]):
    strategy_id = 'gen_local_1790970902'

    @classmethod
    def params_type(cls):
        return WaveletParams

    @staticmethod
    def warmup_bars(params):
        # Eight gap observations require nine prices; confirmation adds one.
        return 10

    @staticmethod
    def indicators(data, params):
        if not (-1.0 <= params.coherence_floor <= 1.0):
            raise ValueError('coherence_floor must be between -1 and 1')
        if not np.isfinite(params.drift_floor):
            raise ValueError('drift_floor must be finite')
        out = pd.DataFrame(index=data.index)
        close = data['close'].where(data['close'] > 0)
        opening = data['open'].where(data['open'] > 0)
        out['gap'] = np.log(opening / close.shift(1)).replace([np.inf, -np.inf], np.nan)
        out['session'] = np.log(close / opening).replace([np.inf, -np.inf], np.nan)
        out['daily_return'] = out['gap'] + out['session']
        out['drift'] = out['daily_return'].rolling(8, min_periods=8).mean()
        gap_energy = pd.Series(0.0, index=data.index)
        session_energy = pd.Series(0.0, index=data.index)
        cross_energy = pd.Series(0.0, index=data.index)
        # Complete trailing eight-point orthonormal Haar detail decomposition.
        for scale in (2, 4, 8):
            half = scale // 2
            for name in ('gap', 'session'):
                sums = out[name].rolling(half, min_periods=half).sum()
                out[f'{name}_haar_{scale}'] = (sums - sums.shift(half)) / np.sqrt(scale)
            g = out[f'gap_haar_{scale}']
            s = out[f'session_haar_{scale}']
            eg = pd.Series(0.0, index=data.index)
            es = pd.Series(0.0, index=data.index)
            cross = pd.Series(0.0, index=data.index)
            for offset in range(0, 8, scale):
                eg = eg + g.shift(offset).pow(2)
                es = es + s.shift(offset).pow(2)
                cross = cross + g.shift(offset) * s.shift(offset)
            out[f'gap_energy_{scale}'] = eg
            out[f'session_energy_{scale}'] = es
            out[f'cross_energy_{scale}'] = cross
            gap_energy = gap_energy + eg
            session_energy = session_energy + es
            cross_energy = cross_energy + cross
        out['gap_energy'] = gap_energy
        out['session_energy'] = session_energy
        out['cross_energy'] = cross_energy
        out['total_energy'] = gap_energy + session_energy
        out['coherence'] = (2.0 * cross_energy / out['total_energy'].where(out['total_energy'] > 1e-16)).clip(-1.0, 1.0)
        valid = out[['coherence', 'drift']].notna().all(axis=1)
        eligible = (out['coherence'] >= params.coherence_floor) & (out['drift'] > params.drift_floor)
        out['eligible'] = eligible.astype(float).where(valid)
        out['confirmation'] = out['eligible'].rolling(2, min_periods=2).sum()
        return out

    @staticmethod
    def generate_signals(data, indicators, ctx, params):
        # The entry predicate flips only after two consecutive agreeing bars.
        # Mixed or unavailable observations preserve the last confirmed regime.
        confirmation = indicators['confirmation']
        flips = pd.Series(np.nan, index=data.index, dtype=float)
        flips.loc[confirmation.eq(2.0)] = 1.0
        flips.loc[confirmation.eq(0.0)] = 0.0
        df = pd.DataFrame(index=data.index)
        df['signal'] = flips.ffill().fillna(0).astype(int)
        df['size'] = 0.95
        df['signal'] = df['signal'].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column='signal', size_column='size')


def _self_check():
    params = WaveletParams()
    index = pd.date_range('2020-01-01', periods=64, freq='B')
    x = 0.001 + 0.0005 * np.sin(np.arange(64))
    closing = 100.0 * np.exp(np.cumsum(2.0 * x))
    opening = closing / np.exp(x)
    data = pd.DataFrame({'open': opening, 'high': np.maximum(opening, closing) * 1.001, 'low': np.minimum(opening, closing) * 0.999, 'close': closing, 'volume': 1000000.0}, index=index)
    features = GeneratedStrategy.indicators(data, params)
    signals = GeneratedStrategy.generate_signals(data, features, None, params).data
    assert GeneratedStrategy.warmup_bars(params) == 10
    assert np.allclose(features['coherence'].iloc[8:], 1.0)
    assert signals['signal'].iloc[:10].eq(0).all()
    assert signals['signal'].iloc[10:].eq(1).all()
    prefix = data.iloc[:32]
    prefix_features = GeneratedStrategy.indicators(prefix, params)
    prefix_signals = GeneratedStrategy.generate_signals(prefix, prefix_features, None, params).data
    pd.testing.assert_frame_equal(signals.iloc[:32], prefix_signals)
    reversal = features.copy()
    reversal.loc[index[20]:, 'confirmation'] = 0.0
    exited = GeneratedStrategy.generate_signals(data, reversal, None, params).data
    assert exited['signal'].iloc[20] == 1
    assert exited['signal'].iloc[21:].eq(0).all()
    flat = data.copy()
    flat[['open', 'high', 'low', 'close']] = 100.0
    flat_features = GeneratedStrategy.indicators(flat, params)
    assert GeneratedStrategy.generate_signals(flat, flat_features, None, params).data['signal'].eq(0).all()


if __name__ == '__main__':
    _self_check()
