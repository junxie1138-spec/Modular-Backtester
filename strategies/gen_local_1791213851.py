from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class ShockParams:
    wave_window: int = 3
    baseline_window: int = 40
    atr_window: int = 14
    regime_ratio: float = 1.15
    shock_ratio: float = 1.20
    stop_atr: float = 1.80
    hold_days: int = 4

    def __post_init__(self):
        if self.wave_window < 2 or self.baseline_window < 2 or self.atr_window < 1:
            raise ValueError('Indicator windows must be positive; volatility windows require at least two bars.')
        if not 3 <= self.hold_days <= 5:
            raise ValueError('hold_days must be between three and five.')
        if not all(np.isfinite(x) for x in (self.regime_ratio, self.shock_ratio, self.stop_atr)):
            raise ValueError('Thresholds must be finite.')
        if self.regime_ratio <= 0 or self.shock_ratio <= 1 or self.stop_atr <= 0:
            raise ValueError('Invalid regime, shock, or stop threshold.')


class GeneratedStrategy(BaseStrategy[ShockParams]):
    strategy_id = 'gen_local_1791213851'

    @classmethod
    def params_type(cls):
        return ShockParams

    def warmup_bars(self, params: ShockParams) -> int:
        return int(max(params.baseline_window, 2 * params.wave_window, params.atr_window) + 1)

    def indicators(self, data: pd.DataFrame, params: ShockParams) -> pd.DataFrame:
        out = pd.DataFrame(index=data.index)
        close = data['close']
        previous = close.shift(1)
        out['return'] = close.pct_change(fill_method=None)
        out['true_range'] = pd.concat([
            data['high'] - data['low'],
            (data['high'] - previous).abs(),
            (data['low'] - previous).abs(),
        ], axis=1).max(axis=1)
        out['atr'] = out['true_range'].rolling(params.atr_window, min_periods=params.atr_window).mean()
        w = params.wave_window
        out['front_vol'] = out['return'].rolling(w, min_periods=w).std(ddof=0)
        out['back_vol'] = out['front_vol'].shift(w)
        out['baseline_vol'] = out['return'].rolling(params.baseline_window, min_periods=params.baseline_window).std(ddof=0)
        out['regime'] = out['front_vol'] / out['baseline_vol'].replace(0.0, np.nan)
        out['wave'] = out['front_vol'] / out['back_vol'].replace(0.0, np.nan)
        out['impulse'] = out['return'].rolling(w, min_periods=w).sum()
        out['previous_impulse'] = out['impulse'].shift(1)
        out['dissipation'] = 1.0 / out['wave'].replace(0.0, np.nan)
        ready = np.isfinite(out[['atr', 'regime', 'wave', 'impulse', 'previous_impulse', 'dissipation']]).all(axis=1) & out['atr'].gt(0)
        calm = out['regime'].le(params.regime_ratio)
        forming = calm & out['wave'].ge(params.shock_ratio) & out['impulse'].gt(0)
        dissipating = (~calm) & out['dissipation'].ge(params.shock_ratio) & out['previous_impulse'].lt(0)
        out['entry'] = (ready & out['return'].gt(0) & (forming | dissipating)).fillna(False)
        out['strength'] = pd.Series(np.where(calm, out['wave'], out['dissipation']), index=data.index)
        out['strength'] = ((out['strength'] - 1.0) / (2.0 * (params.shock_ratio - 1.0))).replace([np.inf, -np.inf], np.nan).fillna(0.0).clip(0.0, 1.0)
        return out

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: ShockParams) -> SignalFrame:
        n = len(data)
        signals = np.zeros(n, dtype=int)
        sizes = np.ones(n, dtype=float)
        entries = indicators['entry'].fillna(False).to_numpy(dtype=bool)
        atr = indicators['atr'].to_numpy(dtype=float)
        strength = indicators['strength'].to_numpy(dtype=float)
        opens = data['open'].to_numpy(dtype=float)
        closes = data['close'].to_numpy(dtype=float)
        active = False
        entry_bar = -1
        frozen_atr = 0.0
        stop_price = np.nan
        exposure = 1.0
        for i in range(n):
            if active:
                if i == entry_bar:
                    # The prior close's instruction fills at this bar's open.
                    stop_price = opens[i] - params.stop_atr * frozen_atr
                expired = i - entry_bar + 1 >= params.hold_days
                stopped = not np.isfinite(closes[i]) or not np.isfinite(stop_price) or closes[i] < stop_price
                if expired or stopped:
                    active = False
                else:
                    signals[i] = 1
                    sizes[i] = exposure
            elif entries[i] and np.isfinite(atr[i]) and atr[i] > 0 and np.isfinite(strength[i]):
                active = True
                entry_bar = i + 1
                frozen_atr = atr[i]
                exposure = float(0.25 + 0.75 * np.clip(strength[i], 0.0, 1.0))
                signals[i] = 1
                sizes[i] = exposure
        df = pd.DataFrame({'signal': signals, 'size': sizes}, index=data.index)
        df['signal'] = df['signal'].shift(1).fillna(0).astype(int)
        df['size'] = df['size'].shift(1).fillna(1.0).astype(float)
        return SignalFrame(data=df, signal_column='signal', size_column='size')


def _self_check():
    index = pd.date_range('2020-01-01', periods=11, freq='D')
    data = pd.DataFrame({'open': 100.0, 'high': 131.0, 'low': 96.0, 'close': [100., 130., 101., 97., 100., 130., 101., 101., 101., 101., 101.], 'volume': 1000.0}, index=index)
    indicators = pd.DataFrame({'entry': [False, True, False, False, False, True, False, False, False, False, False], 'atr': [2., 2., 100., 100., 2., 2., 100., 100., 100., 100., 100.], 'strength': [0., 0.2, 0., 0., 0., 0.8, 0., 0., 0., 0., 0.]}, index=index)
    strategy = GeneratedStrategy()
    params = ShockParams(stop_atr=1.0)
    result = strategy.generate_signals(data, indicators, None, params).data
    assert result['signal'].tolist() == [0, 0, 1, 1, 0, 0, 1, 1, 1, 1, 0]
    assert np.isclose(result['size'].iloc[2], 0.4)
    assert np.isclose(result['size'].iloc[6], 0.85)
    assert result['size'].gt(0).all()
    computed = strategy.indicators(data, params)
    assert computed.index.equals(data.index)
    assert not computed['entry'].any()
    assert strategy.warmup_bars(params) == 41


if __name__ == '__main__':
    _self_check()
