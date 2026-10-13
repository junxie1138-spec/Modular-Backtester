from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class RecoveryParams:
    peak_window: int = 63
    gap_window: int = 6
    pressure_window: int = 5
    atr_window: int = 14
    min_drawdown: float = 0.04
    max_drawdown: float = 0.20
    gap_trigger: float = 0.15
    gap_debt: float = 0.50
    trail_atr: float = 3.0
    hold_bars: int = 18

    def __post_init__(self):
        if min(self.peak_window, self.gap_window, self.pressure_window, self.atr_window, self.hold_bars) < 1:
            raise ValueError('Lookbacks and holding period must be positive.')
        values = (self.min_drawdown, self.max_drawdown, self.gap_trigger, self.gap_debt, self.trail_atr)
        if not all(np.isfinite(v) for v in values):
            raise ValueError('Thresholds must be finite.')
        if not 0 < self.min_drawdown < self.max_drawdown < 1:
            raise ValueError('Drawdown bounds must satisfy 0 < minimum < maximum < 1.')
        if min(self.gap_trigger, self.gap_debt, self.trail_atr) <= 0:
            raise ValueError('Gap thresholds and trailing multiplier must be positive.')


class GeneratedStrategy(BaseStrategy[RecoveryParams]):
    strategy_id = 'gen_local-w2_1791222413'

    @classmethod
    def params_type(cls):
        return RecoveryParams

    def warmup_bars(self, params: RecoveryParams) -> int:
        return int(max(params.peak_window + 1, params.gap_window + 2,
                       params.pressure_window + 1, params.atr_window + 2))

    def indicators(self, data: pd.DataFrame, params: RecoveryParams) -> pd.DataFrame:
        out = pd.DataFrame(index=data.index)
        out['previous_close'] = data['close'].shift(1)
        out['gap'] = data['open'] - out['previous_close']
        out['range'] = data['high'] - data['low']
        out['true_range'] = pd.concat([
            out['range'],
            (data['high'] - out['previous_close']).abs(),
            (data['low'] - out['previous_close']).abs(),
        ], axis=1).max(axis=1)
        out['atr'] = out['true_range'].rolling(params.atr_window, min_periods=params.atr_window).mean()
        out['previous_atr'] = out['atr'].shift(1).replace(0.0, np.nan)
        out['gap_impulse'] = out['gap'] / out['previous_atr']
        out['prior_gap_sum'] = out['gap'].shift(1).rolling(params.gap_window, min_periods=params.gap_window).sum()
        out['gap_reservoir'] = out['prior_gap_sum'] / out['previous_atr']
        out['prior_peak'] = data['close'].shift(1).rolling(params.peak_window, min_periods=params.peak_window).max()
        out['drawdown'] = 1.0 - data['close'] / out['prior_peak'].replace(0.0, np.nan)
        out['close_location'] = ((2.0 * data['close'] - data['high'] - data['low']) / out['range'].replace(0.0, np.nan)).clip(-1.0, 1.0)
        out.loc[out['range'].eq(0.0), 'close_location'] = 0.0
        out['valid_volume'] = data['volume'].where(data['volume'] >= 0.0)
        out['signed_volume'] = out['close_location'] * out['valid_volume']
        out['pressure_numerator'] = out['signed_volume'].rolling(params.pressure_window, min_periods=params.pressure_window).sum()
        out['pressure_denominator'] = out['valid_volume'].rolling(params.pressure_window, min_periods=params.pressure_window).sum()
        out['pressure'] = out['pressure_numerator'] / out['pressure_denominator'].replace(0.0, np.nan)
        out['previous_pressure'] = out['pressure'].shift(1)
        return out.replace([np.inf, -np.inf], np.nan)

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame,
                         ctx: StrategyContext, params: RecoveryParams) -> SignalFrame:
        gap_agrees = (indicators['gap_impulse'] >= params.gap_trigger) & (indicators['gap_reservoir'] <= -params.gap_debt)
        pressure_agrees = (indicators['pressure'] > 0.0) & (indicators['previous_pressure'] <= 0.0)
        eligible = indicators['drawdown'].between(params.min_drawdown, params.max_drawdown)
        required = ['gap_impulse', 'gap_reservoir', 'pressure', 'previous_pressure', 'drawdown', 'atr']
        ready = indicators[required].notna().all(axis=1) & (indicators['atr'] > 0.0)
        entries = (gap_agrees & pressure_agrees & eligible & ready).fillna(False).to_numpy(dtype=bool)
        closes = data['close'].to_numpy(dtype=float)
        atrs = indicators['atr'].to_numpy(dtype=float)
        signals = np.zeros(len(data), dtype=int)
        active = False
        age = 0
        high_water = np.nan
        stop = -np.inf
        warmup = self.warmup_bars(params)

        for i in range(len(data)):
            if active:
                # Entry was ordered at the previous close and filled at this bar's open.
                age += 1
                if not np.isfinite(closes[i]) or not np.isfinite(atrs[i]) or atrs[i] <= 0.0:
                    active = False
                    continue
                high_water = closes[i] if age == 1 else max(high_water, closes[i])
                stop = max(stop, high_water - params.trail_atr * atrs[i])
                if closes[i] <= stop or age >= params.hold_bars:
                    active = False
                else:
                    signals[i] = 1
            elif i >= warmup and entries[i] and np.isfinite(closes[i]) and closes[i] > 0.0:
                active = True
                age = 0
                high_water = np.nan
                stop = -np.inf
                signals[i] = 1

        df = pd.DataFrame({'signal': signals, 'size': 1.0}, index=data.index)
        df['signal'] = df['signal'].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column='signal', size_column='size')


def _self_check():
    params = RecoveryParams()
    strategy = GeneratedStrategy.__new__(GeneratedStrategy)
    w = strategy.warmup_bars(params)
    n = w + params.hold_bars + 6
    index = pd.date_range('2020-01-01', periods=n, freq='B')
    data = pd.DataFrame({'open': 100.0, 'high': 101.0, 'low': 99.0, 'close': 100.0, 'volume': 1000.0}, index=index)
    features = pd.DataFrame({'gap_impulse': 0.0, 'gap_reservoir': -1.0, 'pressure': 0.2, 'previous_pressure': -0.2, 'drawdown': 0.08, 'atr': 1.0}, index=index)
    features.loc[index[w], 'gap_impulse'] = 0.3
    data.loc[index[w + 2], 'close'] = 110.0
    data.loc[index[w + 3], 'close'] = 108.0
    features.loc[index[w + 3], 'atr'] = 10.0
    data.loc[index[w + 4], 'close'] = 106.0
    result = strategy.generate_signals(data, features, None, params).data
    assert result['signal'].iloc[w] == 0
    assert result['signal'].iloc[w + 1:w + 5].eq(1).all()
    assert result['signal'].iloc[w + 5] == 0
    data['close'] = 100.0
    features['atr'] = 1.0
    result = strategy.generate_signals(data, features, None, params).data
    assert result['signal'].iloc[w + 1:w + params.hold_bars + 1].eq(1).all()
    assert result['signal'].iloc[w + params.hold_bars + 1] == 0
    features['pressure'] = -0.2
    assert strategy.generate_signals(data, features, None, params).data['signal'].eq(0).all()
    assert strategy.indicators(data, params).index.equals(data.index)


if __name__ == '__main__':
    _self_check()
