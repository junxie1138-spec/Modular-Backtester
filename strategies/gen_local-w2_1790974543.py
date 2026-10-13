from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class RangeElasticityParams:
    learning_bars: int = 126
    entry_bps: float = 6.0

    def __post_init__(self):
        if self.learning_bars < 20:
            raise ValueError('learning_bars must be at least 20')
        if not np.isfinite(self.entry_bps) or self.entry_bps <= 0:
            raise ValueError('entry_bps must be finite and positive')


class GeneratedStrategy(BaseStrategy[RangeElasticityParams]):
    strategy_id = 'gen_local-w2_1790974543'

    @classmethod
    def params_type(cls) -> type[RangeElasticityParams]:
        return RangeElasticityParams

    def warmup_bars(self, params: RangeElasticityParams) -> int:
        return int(params.learning_bars + 22)

    def indicators(self, data: pd.DataFrame, params: RangeElasticityParams) -> pd.DataFrame:
        out = pd.DataFrame(index=data.index)
        out['weekday'] = data.index.dayofweek
        out['range'] = (data['high'] - data['low']).where(data['high'] > data['low'])
        out['relative_range'] = out['range'] / data['close'].where(data['close'] > 0)
        out['range_baseline'] = out['relative_range'].rolling(20, min_periods=20).mean().shift(1)
        out['expansion'] = (out['relative_range'] / out['range_baseline'] - 1.0).clip(lower=0.0)
        out['body_fraction'] = ((data['close'] - data['open']) / out['range']).clip(-1.0, 1.0)
        out['pulse'] = out['expansion'] * out['body_fraction']
        out['completed_return'] = data['close'].pct_change(periods=2, fill_method=None)
        out['matured_pulse'] = out['pulse'].shift(2)
        out['matured_weekday'] = out['weekday'].shift(2)
        out['forecast'] = np.nan
        x = out['matured_pulse']
        y = out['completed_return']
        valid = np.isfinite(x) & np.isfinite(y) & x.ne(0.0)
        for weekday in range(5):
            mask = valid & out['matured_weekday'].eq(weekday)
            prefix = f'weekday_{weekday}'
            out[prefix + '_count'] = mask.astype(float).rolling(params.learning_bars, min_periods=params.learning_bars).sum()
            out[prefix + '_cross'] = (x * y).where(mask, 0.0).rolling(params.learning_bars, min_periods=params.learning_bars).sum()
            out[prefix + '_square'] = x.pow(2).where(mask, 0.0).rolling(params.learning_bars, min_periods=params.learning_bars).sum()
            denominator = out[prefix + '_square'].where(out[prefix + '_square'] > 0.0)
            out[prefix + '_coefficient'] = (out[prefix + '_cross'] / denominator).where(out[prefix + '_count'] >= 8.0)
            today = out['weekday'].eq(weekday)
            out.loc[today, 'forecast'] = out.loc[today, prefix + '_coefficient'] * out.loc[today, 'pulse']
        return out.replace([np.inf, -np.inf], np.nan)

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: RangeElasticityParams) -> SignalFrame:
        forecast = indicators['forecast'].to_numpy(dtype=float)
        signal = np.zeros(len(data), dtype=int)
        position = 0
        exit_bar = -1
        threshold = params.entry_bps / 10000.0
        for i, value in enumerate(forecast):
            if position:
                if i == exit_bar:
                    position = 0
            elif np.isfinite(value) and abs(value) >= threshold:
                position = 1 if value > 0.0 else -1
                exit_bar = i + 2
            signal[i] = position
        df = pd.DataFrame({'signal': signal, 'size': 1.0}, index=data.index)
        df['signal'] = df['signal'].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column='signal', size_column='size')


if __name__ == '__main__':
    index = pd.date_range('2020-01-01', periods=8, freq='B')
    data = pd.DataFrame(index=index)
    indicators = pd.DataFrame({'forecast': [0.01, -0.01, -0.01, -0.01, 0.01, 0.01, 0.01, 0.0]}, index=index)
    result = GeneratedStrategy.generate_signals(None, data, indicators, None, RangeElasticityParams())
    assert result.data['signal'].tolist() == [0, 1, 1, 0, -1, -1, 0, 1]
    assert result.data['size'].gt(0.0).all()
