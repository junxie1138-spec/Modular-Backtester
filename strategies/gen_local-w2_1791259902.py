from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class Params:
    lookback: int = 12
    noise_window: int = 20
    shock_z: float = 1.5
    concentration: float = 0.45
    recovery_min: float = 0.15
    recovery_max: float = 0.65
    hold_bars: int = 2


class GeneratedStrategy(BaseStrategy[Params]):
    strategy_id = 'gen_local-w2_1791259902'

    @classmethod
    def params_type(cls) -> type[Params]:
        return Params

    def warmup_bars(self, params: Params) -> int:
        return int(max(params.lookback, params.noise_window) + 3)

    def indicators(self, data: pd.DataFrame, params: Params) -> pd.DataFrame:
        if params.lookback < 2 or params.noise_window < 2:
            raise ValueError('Lookback windows must be at least two.')
        if params.hold_bars not in (1, 2):
            raise ValueError('hold_bars must be one or two.')
        if not (0 < params.concentration <= 1 and params.shock_z > 0):
            raise ValueError('Invalid shock thresholds.')
        if not (0 < params.recovery_min < params.recovery_max < 1):
            raise ValueError('Recovery fractions must lie strictly between zero and one.')

        out = pd.DataFrame(index=data.index)
        close = data['close'].where(data['close'] > 0)
        out['return'] = np.log(close).diff().replace([np.inf, -np.inf], np.nan)
        r = out['return']
        loss = (-r).clip(lower=0)
        gain = r.clip(lower=0)
        out['loss'] = loss
        out['gain'] = gain
        # Freeze the dislocation window before the two confirmation bars.
        out['net_move'] = r.rolling(params.lookback, min_periods=params.lookback).sum().shift(2)
        out['loss_total'] = loss.rolling(params.lookback, min_periods=params.lookback).sum().shift(2)
        out['gain_total'] = gain.rolling(params.lookback, min_periods=params.lookback).sum().shift(2)
        out['largest_loss'] = loss.rolling(params.lookback, min_periods=params.lookback).max().shift(2)
        out['largest_gain'] = gain.rolling(params.lookback, min_periods=params.lookback).max().shift(2)
        out['noise'] = r.rolling(params.noise_window, min_periods=params.noise_window).std(ddof=0).shift(2)
        out['shock_scale'] = out['noise'] * np.sqrt(params.lookback)
        out['shock_strength'] = out['net_move'].abs().div(out['shock_scale'].where(out['shock_scale'] > 0))
        out['loss_concentration'] = out['largest_loss'].div(out['loss_total'].where(out['loss_total'] > 0))
        out['gain_concentration'] = out['largest_gain'].div(out['gain_total'].where(out['gain_total'] > 0))
        out['confirmation_move'] = r + r.shift(1)
        out['recovery_fraction'] = -out['confirmation_move'].div(out['net_move'].where(out['net_move'] != 0))
        recovered = out['recovery_fraction'].between(params.recovery_min, params.recovery_max)
        strong = out['shock_strength'].ge(params.shock_z)
        long_entry = (out['net_move'].lt(0) & strong & recovered
                      & out['loss_concentration'].ge(params.concentration)
                      & r.gt(0) & r.shift(1).gt(0))
        short_entry = (out['net_move'].gt(0) & strong & recovered
                       & out['gain_concentration'].ge(params.concentration)
                       & r.lt(0) & r.shift(1).lt(0))
        out['entry'] = np.select([long_entry.fillna(False), short_entry.fillna(False)], [1, -1], default=0).astype(int)
        out.iloc[:self.warmup_bars(params), out.columns.get_loc('entry')] = 0
        return out

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame,
                         ctx: StrategyContext, params: Params) -> SignalFrame:
        if params.hold_bars not in (1, 2):
            raise ValueError('hold_bars must be one or two.')
        entries = indicators['entry'].fillna(0).to_numpy(dtype=int)
        signals = np.zeros(len(data), dtype=int)
        position = 0
        exit_at = -1
        for i, candidate in enumerate(entries):
            if position:
                if i >= exit_at:
                    position = 0
                    continue
                signals[i] = position
            elif candidate:
                position = int(candidate)
                exit_at = i + params.hold_bars
                signals[i] = position
        df = pd.DataFrame({'signal': signals, 'size': 0.95}, index=data.index)
        df['signal'] = df['signal'].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column='signal', size_column='size')


def _self_check() -> None:
    strategy = GeneratedStrategy()
    params = Params()
    index = pd.date_range('2020-01-01', periods=8)
    data = pd.DataFrame(index=index)
    indicators = pd.DataFrame({'entry': [1, -1, -1, -1, 1, 1, 1, 0]}, index=index)
    result = strategy.generate_signals(data, indicators, None, params).data
    assert result['signal'].tolist() == [0, 1, 1, 0, -1, -1, 0, 1]
    assert result['size'].gt(0).all()
    params.hold_bars = 1
    result = strategy.generate_signals(data, indicators, None, params).data
    assert result['signal'].tolist() == [0, 1, 0, -1, 0, 1, 0, 1]


if __name__ == '__main__':
    _self_check()
