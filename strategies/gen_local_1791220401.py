from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class OscillationParams:
    ma_window: int = 30
    half_cycle: int = 8
    correlation_window: int = 64
    percentile_window: int = 126
    entry_percentile: float = 0.85
    atr_window: int = 14
    trailing_atr: float = 2.5
    holding_bars: int = 18

    def __post_init__(self):
        if min(self.ma_window, self.correlation_window, self.percentile_window, self.atr_window) < 2:
            raise ValueError('Rolling windows must be at least two bars')
        if self.half_cycle < 1 or self.holding_bars < 1:
            raise ValueError('Cycle lag and holding period must be positive')
        if not 0.0 < self.entry_percentile < 1.0:
            raise ValueError('Entry percentile must lie between zero and one')
        if not np.isfinite(self.trailing_atr) or self.trailing_atr <= 0:
            raise ValueError('ATR multiplier must be finite and positive')


class GeneratedStrategy(BaseStrategy[OscillationParams]):
    strategy_id = 'gen_local_1791220401'

    @classmethod
    def params_type(cls):
        return OscillationParams

    def warmup_bars(self, params):
        return int(max(
            params.ma_window + 2 * params.half_cycle + params.correlation_window - 1,
            params.ma_window + params.percentile_window,
            params.ma_window + 1,
            params.atr_window + 1,
        ))

    def indicators(self, data, params):
        out = pd.DataFrame(index=data.index)
        close = data['close'].astype(float)
        out['ma'] = close.rolling(params.ma_window).mean()
        out['std'] = close.rolling(params.ma_window).std(ddof=0)
        out['z'] = (close - out['ma']) / out['std'].replace(0.0, np.nan)
        out['z'] = out['z'].replace([np.inf, -np.inf], np.nan)
        out['half_lag'] = out['z'].shift(params.half_cycle)
        out['full_lag'] = out['z'].shift(2 * params.half_cycle)
        out['half_corr'] = out['z'].rolling(params.correlation_window).corr(out['half_lag'])
        out['full_corr'] = out['z'].rolling(params.correlation_window).corr(out['full_lag'])
        out['z_change'] = out['z'].diff()
        out['tail_threshold'] = out['z'].abs().rolling(params.percentile_window).quantile(params.entry_percentile).shift(1)
        out['previous_close'] = close.shift(1)
        out['true_range'] = pd.concat([
            data['high'] - data['low'],
            (data['high'] - out['previous_close']).abs(),
            (data['low'] - out['previous_close']).abs(),
        ], axis=1).max(axis=1)
        out.loc[out['previous_close'].isna(), 'true_range'] = np.nan
        out['atr'] = out['true_range'].rolling(params.atr_window).mean()
        return out.replace([np.inf, -np.inf], np.nan)

    def generate_signals(self, data, indicators, ctx: StrategyContext, params):
        df = pd.DataFrame(index=data.index)
        columns = ['z', 'z_change', 'tail_threshold', 'half_corr', 'full_corr', 'atr']
        ready = pd.Series(np.isfinite(indicators[columns].to_numpy()).all(axis=1), index=data.index)
        ready &= indicators['atr'].gt(0) & indicators['tail_threshold'].gt(0)
        oscillating = indicators['half_corr'].lt(0) & indicators['full_corr'].gt(0)
        extreme = indicators['z'].abs().ge(indicators['tail_threshold'])
        long_entry = ready & oscillating & extreme & indicators['z'].lt(0) & indicators['z_change'].gt(0)
        short_entry = ready & oscillating & extreme & indicators['z'].gt(0) & indicators['z_change'].lt(0)
        entries = np.select([long_entry, short_entry], [1, -1], default=0)
        closes = data['close'].to_numpy(dtype=float)
        atrs = indicators['atr'].to_numpy(dtype=float)
        signals = np.zeros(len(data), dtype=int)
        position = 0
        entry_bar = 0
        high_water = low_water = stop = 0.0

        for i in range(len(data)):
            close, atr = closes[i], atrs[i]
            if not np.isfinite(close) or not np.isfinite(atr) or atr <= 0:
                position = 0
                continue
            if position:
                high_water = max(high_water, close)
                low_water = min(low_water, close)
                if position == 1:
                    stop = max(stop, high_water - params.trailing_atr * atr)
                    stopped = close <= stop
                else:
                    # Mirror the highest-close trailing stop for short exposure.
                    stop = min(stop, low_water + params.trailing_atr * atr)
                    stopped = close >= stop
                if stopped or i - entry_bar >= params.holding_bars:
                    position = 0
                    continue
            elif entries[i]:
                position = int(entries[i])
                entry_bar = i
                high_water = low_water = close
                stop = close - position * params.trailing_atr * atr
            signals[i] = position

        df['signal'] = signals
        df['signal'] = df['signal'].shift(1).fillna(0).astype(int)
        df['size'] = 0.95
        return SignalFrame(data=df, signal_column='signal', size_column='size')


def _self_check():
    index = pd.date_range('2020-01-01', periods=8)
    data = pd.DataFrame({'close': [100, 105, 104, 100, 100, 95, 101, 100]}, index=index)
    indicators = pd.DataFrame({
        'z': [-3, 0, 0, 0, 3, 0, 0, 0],
        'z_change': [1, 0, 0, 0, -1, 0, 0, 0],
        'tail_threshold': 2.0,
        'half_corr': -0.5,
        'full_corr': 0.5,
        'atr': [1, 1, 5, 1, 1, 1, 1, 1],
    }, index=index)
    strategy = GeneratedStrategy.__new__(GeneratedStrategy)
    result = strategy.generate_signals(data, indicators, None, OscillationParams()).data
    assert result['signal'].tolist() == [0, 1, 1, 1, 0, -1, -1, 0]
    assert result['size'].gt(0).all()
    expired = strategy.generate_signals(data, indicators, None, OscillationParams(holding_bars=2)).data
    assert expired['signal'].tolist() == [0, 1, 1, 0, 0, -1, -1, 0]
    indicators.loc[:, 'z'] = np.nan
    assert strategy.generate_signals(data, indicators, None, OscillationParams()).data['signal'].eq(0).all()


if __name__ == '__main__':
    _self_check()
