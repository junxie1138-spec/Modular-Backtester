from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class StrategyParams:
    calibration_bars: int = 126
    streak_percentile: float = 0.90
    range_percentile: float = 0.40
    displacement_bars: int = 5
    hold_bars: int = 18

    def __post_init__(self):
        if self.calibration_bars < 2 or self.displacement_bars < 1 or self.hold_bars < 1:
            raise ValueError('Lookbacks and holding period must be positive.')
        if not 0.0 < self.streak_percentile < 1.0:
            raise ValueError('streak_percentile must be between zero and one.')
        if not 0.0 < self.range_percentile < 1.0:
            raise ValueError('range_percentile must be between zero and one.')


class GeneratedStrategy(BaseStrategy[StrategyParams]):
    strategy_id = 'gen_local-w2_1791308789'

    @classmethod
    def params_type(cls):
        return StrategyParams

    @staticmethod
    def warmup_bars(params: StrategyParams) -> int:
        return int(max(params.calibration_bars + 1, params.displacement_bars + 1))

    @staticmethod
    def indicators(data: pd.DataFrame, params: StrategyParams) -> pd.DataFrame:
        out = pd.DataFrame(index=data.index)
        close = data['close'].where(np.isfinite(data['close']) & data['close'].gt(0))
        high = data['high'].where(np.isfinite(data['high']))
        low = data['low'].where(np.isfinite(data['low']))
        out['midpoint'] = (high + low) / 2.0
        out['weak_close'] = (close.lt(out['midpoint']) & high.gt(low) & close.ge(low)).fillna(False)
        weak = out['weak_close']
        out['streak'] = weak.astype(int).groupby((~weak).cumsum()).cumsum().clip(upper=params.calibration_bars)
        out['range_fraction'] = ((high - low) / close).where(high.gt(low)).replace([np.inf, -np.inf], np.nan)
        out['displacement'] = close.diff(params.displacement_bars)
        lookback = params.calibration_bars
        out['streak_threshold'] = out['streak'].shift(1).rolling(lookback, min_periods=lookback).quantile(params.streak_percentile)
        out['range_threshold'] = out['range_fraction'].shift(1).rolling(lookback, min_periods=lookback).quantile(params.range_percentile)
        return out

    @staticmethod
    def generate_signals(data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: StrategyParams) -> SignalFrame:
        eligible = (
            indicators['streak'].gt(indicators['streak_threshold'])
            & indicators['range_fraction'].le(indicators['range_threshold'])
            & indicators['displacement'].lt(0)
        ).fillna(False).to_numpy(dtype=bool)
        signals = np.zeros(len(data), dtype=np.int64)
        exit_bar = -1
        for i, enter in enumerate(eligible):
            if exit_bar >= 0:
                if i < exit_bar:
                    signals[i] = 1
                else:
                    exit_bar = -1
            elif enter:
                signals[i] = 1
                exit_bar = i + params.hold_bars
        df = pd.DataFrame({'signal': signals, 'size': 0.95}, index=data.index)
        df['signal'] = df['signal'].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column='signal', size_column='size')


def _self_check():
    params = StrategyParams()
    index = pd.date_range('2020-01-01', periods=50, freq='B')
    data = pd.DataFrame(index=index)
    marks = pd.DataFrame({
        'streak': 4.0,
        'streak_threshold': 3.0,
        'range_fraction': 0.01,
        'range_threshold': 0.02,
        'displacement': -1.0,
    }, index=index)
    result = GeneratedStrategy.generate_signals(data, marks, None, params).data
    assert result['signal'].iloc[0] == 0
    assert result['signal'].iloc[1:19].eq(1).all()
    assert result['signal'].iloc[19] == 0
    assert result['signal'].iloc[20:38].eq(1).all()
    assert result['signal'].iloc[38] == 0
    assert result['size'].gt(0).all()
    marks.loc[:, :] = np.nan
    assert GeneratedStrategy.generate_signals(data, marks, None, params).data['signal'].eq(0).all()
    candles = pd.DataFrame({
        'open': [100.0] * 6,
        'high': [102.0] * 6,
        'low': [98.0] * 6,
        'close': [99.0, 99.0, 101.0, 99.0, 99.0, 99.0],
        'volume': [1000.0] * 6,
    }, index=index[:6])
    assert GeneratedStrategy.indicators(candles, params)['streak'].tolist() == [1, 2, 0, 1, 2, 3]


if __name__ == '__main__':
    _self_check()
