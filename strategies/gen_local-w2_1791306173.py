from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class Params:
    anchor_window: int = 10
    calibration_window: int = 126
    volume_window: int = 20
    depth_percentile: float = 0.85
    volume_percentile: float = 0.30
    holding_bars: int = 2

    def __post_init__(self):
        if min(self.anchor_window, self.calibration_window, self.volume_window) < 2:
            raise ValueError('Lookback windows must be at least two bars.')
        if not (0.0 < self.depth_percentile < 1.0 and 0.0 < self.volume_percentile < 1.0):
            raise ValueError('Percentiles must be strictly between zero and one.')
        if self.holding_bars not in (1, 2):
            raise ValueError('Holding period must be one or two bars.')


class GeneratedStrategy(BaseStrategy[Params]):
    strategy_id = 'gen_local-w2_1791306173'

    @classmethod
    def params_type(cls):
        return Params

    def warmup_bars(self, params):
        return int(max(params.anchor_window, params.volume_window) + params.calibration_window)

    def indicators(self, data, params):
        out = pd.DataFrame(index=data.index)
        out['peak'] = data['high'].rolling(params.anchor_window, min_periods=params.anchor_window).max()
        out['trough'] = data['low'].rolling(params.anchor_window, min_periods=params.anchor_window).min()
        out['drawdown'] = (1.0 - data['close'] / out['peak'].where(out['peak'] > 0)).clip(lower=0.0)
        out['runup'] = (data['close'] / out['trough'].where(out['trough'] > 0) - 1.0).clip(lower=0.0)
        out['volume_baseline'] = data['volume'].rolling(params.volume_window, min_periods=params.volume_window).median()
        out['relative_volume'] = data['volume'].where(data['volume'] > 0) / out['volume_baseline'].where(out['volume_baseline'] > 0)
        out['drawdown_threshold'] = out['drawdown'].shift(1).rolling(params.calibration_window, min_periods=params.calibration_window).quantile(params.depth_percentile)
        out['runup_threshold'] = out['runup'].shift(1).rolling(params.calibration_window, min_periods=params.calibration_window).quantile(params.depth_percentile)
        out['volume_threshold'] = out['relative_volume'].shift(1).rolling(params.calibration_window, min_periods=params.calibration_window).quantile(params.volume_percentile)
        return out.replace([np.inf, -np.inf], np.nan)

    def generate_signals(self, data, indicators, ctx, params):
        required = ['drawdown', 'runup', 'relative_volume', 'drawdown_threshold', 'runup_threshold', 'volume_threshold']
        ready = indicators[required].notna().all(axis=1)
        thin = indicators['relative_volume'] < indicators['volume_threshold']
        long_entry = ready & thin & (indicators['drawdown'] > indicators['drawdown_threshold']) & (data['close'] < data['open'])
        short_entry = ready & thin & (indicators['runup'] > indicators['runup_threshold']) & (data['close'] > data['open'])
        entries = np.select([long_entry, short_entry], [1, -1], default=0)
        targets = np.zeros(len(data), dtype=int)
        position = 0
        remaining = 0
        for i, entry in enumerate(entries):
            if position:
                remaining -= 1
                if remaining == 0:
                    position = 0
            elif entry:
                position = int(entry)
                remaining = params.holding_bars
            targets[i] = position
        df = pd.DataFrame({'signal': targets, 'size': 1.0}, index=data.index)
        df['signal'] = df['signal'].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column='signal', size_column='size')


def _self_check():
    index = pd.date_range('2020-01-01', periods=7)
    data = pd.DataFrame({'open': 100.0, 'close': [99.0, 101.0, 101.0, 101.0, 99.0, 99.0, 99.0]}, index=index)
    derived = pd.DataFrame({'drawdown': 0.2, 'runup': 0.2, 'relative_volume': 0.2, 'drawdown_threshold': 0.1, 'runup_threshold': 0.1, 'volume_threshold': 0.5}, index=index)
    strategy = GeneratedStrategy.__new__(GeneratedStrategy)
    result = strategy.generate_signals(data, derived, None, Params())
    assert result.data['signal'].tolist() == [0, 1, 1, 0, -1, -1, 0]
    result = strategy.generate_signals(data, derived, None, Params(holding_bars=1))
    assert result.data['signal'].tolist() == [0, 1, 0, -1, 0, 1, 0]
    derived.loc[:, 'volume_threshold'] = np.nan
    assert strategy.generate_signals(data, derived, None, Params()).data['signal'].eq(0).all()


if __name__ == '__main__':
    _self_check()
