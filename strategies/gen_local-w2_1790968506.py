from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class SignatureParams:
    signature_window: int = 6
    regime_window: int = 20
    efficiency_cutoff: float = 0.35
    area_threshold: float = 0.06
    atr_window: int = 14
    stop_atr: float = 1.8
    hold_days: int = 4


class GeneratedStrategy(BaseStrategy[SignatureParams]):
    strategy_id = 'gen_local-w2_1790968506'

    @classmethod
    def params_type(cls):
        return SignatureParams

    def warmup_bars(self, params):
        if min(params.signature_window, params.regime_window, params.atr_window) < 2:
            raise ValueError('Lookbacks must be at least two.')
        if not 3 <= params.hold_days <= 5:
            raise ValueError('Holding period must be three to five sessions.')
        if not 0 < params.efficiency_cutoff < 1:
            raise ValueError('Efficiency cutoff must be between zero and one.')
        if not 0 < params.area_threshold < 1:
            raise ValueError('Area threshold must be between zero and one.')
        if not np.isfinite(params.stop_atr) or params.stop_atr <= 0:
            raise ValueError('Stop multiplier must be finite and positive.')
        return int(max(params.signature_window + 2, params.regime_window + 1,
                       params.atr_window + 1) + 1)

    def indicators(self, data, params):
        self.warmup_bars(params)
        out = pd.DataFrame(index=data.index)
        close = data['close'].where(data['close'] > 0)
        previous = close.shift(1)
        out['true_range'] = pd.concat([
            data['high'] - data['low'],
            (data['high'] - previous).abs(),
            (data['low'] - previous).abs()
        ], axis=1).max(axis=1, skipna=False)
        out['atr'] = out['true_range'].rolling(params.atr_window).mean()
        out['price_step'] = np.log(close).diff()
        out['normalized_range'] = out['true_range'] / previous
        out['risk_step'] = np.log(out['normalized_range'].where(
            out['normalized_range'] > 0)).diff()
        out = out.replace([np.inf, -np.inf], np.nan)
        x = out['price_step']
        y = out['risk_step']
        length = params.signature_window
        # Antisymmetric second-order signature: sum over ordered step pairs.
        area = pd.Series(0.0, index=data.index)
        for lag in range(1, length):
            cross = x.shift(lag) * y - y.shift(lag) * x
            area = area + cross.rolling(length - lag).sum()
        out['area_scale'] = x.abs().rolling(length).sum() * y.abs().rolling(length).sum()
        out['area'] = area / out['area_scale'].where(out['area_scale'] > 0)
        out['path_return'] = x.rolling(length).sum()
        out['regime_return'] = x.rolling(params.regime_window).sum()
        out['regime_travel'] = x.abs().rolling(params.regime_window).sum()
        out['efficiency'] = out['regime_return'].abs() / out['regime_travel'].where(
            out['regime_travel'] > 0)
        out['efficient'] = out['efficiency'].ge(params.efficiency_cutoff)
        out['valid'] = out[['area', 'efficiency', 'atr', 'path_return',
                            'price_step']].notna().all(axis=1) & out['atr'].gt(0)
        trend = out['efficient'] & out['regime_return'].gt(0) & out['area'].gt(params.area_threshold)
        recovery = ~out['efficient'] & out['area'].lt(-params.area_threshold)
        out['setup'] = (out['valid'] & out['path_return'].gt(0)
                        & out['price_step'].gt(0) & (trend | recovery))
        out['confirmed'] = (out['setup'] & out['setup'].shift(1, fill_value=False)
                            & out['efficient'].eq(out['efficient'].shift(1)))
        return out

    def generate_signals(self, data, indicators, ctx, params):
        close = data['close'].to_numpy(dtype=float)
        opening = data['open'].to_numpy(dtype=float)
        atr = indicators['atr'].to_numpy(dtype=float)
        confirmed = indicators['confirmed'].fillna(False).to_numpy(dtype=bool)
        decisions = np.zeros(len(data), dtype=int)
        active = False
        entry_bar = -1
        stop = np.nan
        for i in range(len(data)):
            # A prior close's entry decision fills at this session's open.
            if not active and i > 0 and decisions[i - 1] == 1:
                if not (np.isfinite(opening[i]) and opening[i] > 0
                        and np.isfinite(atr[i - 1]) and atr[i - 1] > 0):
                    raise ValueError('Invalid entry open or entry ATR.')
                active = True
                entry_bar = i
                # Frozen entry-open reference and decision-time ATR; never trails.
                stop = opening[i] - params.stop_atr * atr[i - 1]
            if active:
                stop_hit = np.isfinite(close[i]) and close[i] < stop
                expired = i - entry_bar + 1 >= params.hold_days
                if stop_hit or expired:
                    active = False
                    continue
                decisions[i] = 1
            elif confirmed[i] and np.isfinite(atr[i]) and atr[i] > 0:
                decisions[i] = 1
        df = pd.DataFrame({'signal': decisions, 'size': 1.0}, index=data.index)
        df['signal'] = df['signal'].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column='signal', size_column='size')


def _self_check():
    index = pd.date_range('2020-01-01', periods=8)
    data = pd.DataFrame({'open': 100.0, 'high': 112.0, 'low': 96.0,
                         'close': [100., 100., 101., 110., 99., 100., 100., 100.],
                         'volume': 1000.0}, index=index)
    indicators = pd.DataFrame({'atr': 1.0,
                               'confirmed': [False, True, False, False,
                                             False, False, False, False]}, index=index)
    params = SignatureParams()
    run = GeneratedStrategy.generate_signals
    result = run(None, data, indicators, None, params).data
    # Next-open entry, four held sessions, and no trailing stop after 110.
    assert result['signal'].tolist() == [0, 0, 1, 1, 1, 1, 0, 0]
    data.loc[index[3], 'close'] = 97.0
    result = run(None, data, indicators, None, params).data
    # Close breaches fixed 98.2 stop; exit takes effect one bar later.
    assert result['signal'].tolist() == [0, 0, 1, 1, 0, 0, 0, 0]
    assert result['size'].gt(0).all()


if __name__ == '__main__':
    _self_check()
