from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class RangeBreathingParams:
    lookback: int = 15
    breathing_threshold: float = 0.30
    trend_floor: float = 0.04
    hold_bars: int = 8

    def __post_init__(self):
        if self.lookback < 2 or self.hold_bars < 1:
            raise ValueError('lookback must be >= 2 and hold_bars >= 1')
        if not 0.0 < self.breathing_threshold < 1.0:
            raise ValueError('breathing_threshold must be between 0 and 1')
        if not np.isfinite(self.trend_floor) or self.trend_floor < 0.0:
            raise ValueError('trend_floor must be finite and nonnegative')


class GeneratedStrategy(BaseStrategy[RangeBreathingParams]):
    strategy_id = 'gen_local-w2_1791344670'

    @classmethod
    def params_type(cls):
        return RangeBreathingParams

    def warmup_bars(self, params: RangeBreathingParams) -> int:
        return int(params.lookback + 2)

    def indicators(self, data: pd.DataFrame, params: RangeBreathingParams) -> pd.DataFrame:
        out = pd.DataFrame(index=data.index)
        length = params.lookback
        out['range'] = (data['high'] - data['low']).where(data['high'] >= data['low'])
        out['midpoint'] = (data['high'] + data['low']) / 2.0
        out['range_change'] = out['range'].diff()
        out['midpoint_change'] = out['midpoint'].diff()
        out['direction'] = np.sign(out['midpoint_change'])
        out['breathing_impulse'] = out['range_change'] * out['direction']
        out['impulse_sum'] = out['breathing_impulse'].rolling(length, min_periods=length).sum()
        out['range_churn'] = out['range_change'].abs().rolling(length, min_periods=length).sum()
        out['range_sum'] = out['range'].rolling(length, min_periods=length).sum()
        out['midpoint_drift'] = out['midpoint'].diff(length)
        out['breathing'] = out['impulse_sum'] / out['range_churn'].where(out['range_churn'] > 0.0)
        out['trend'] = out['midpoint_drift'] / out['range_sum'].where(out['range_sum'] > 0.0)
        return out.replace([np.inf, -np.inf], np.nan)

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: RangeBreathingParams,
    ) -> SignalFrame:
        breathing = indicators['breathing']
        trend = indicators['trend']
        valid = np.isfinite(breathing) & np.isfinite(trend)
        long_entry = valid & (breathing > params.breathing_threshold) & (trend > params.trend_floor)
        short_entry = valid & (breathing < -params.breathing_threshold) & (trend < -params.trend_floor)
        candidate = pd.Series(np.select([long_entry, short_entry], [1, -1], default=0), index=data.index, dtype=int)
        confirmed = candidate.where(candidate.eq(candidate.shift(1)), 0)
        positions = np.zeros(len(data), dtype=int)
        side = 0
        exit_bar = -1
        for i, entry in enumerate(confirmed.to_numpy()):
            if side and i == exit_bar:
                side = 0
                continue
            if not side and entry:
                side = int(entry)
                exit_bar = i + params.hold_bars
            positions[i] = side
        df = pd.DataFrame({'signal': positions, 'size': 1.0}, index=data.index)
        df['signal'] = df['signal'].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column='signal', size_column='size')


if __name__ == '__main__':
    strategy = GeneratedStrategy()
    params = RangeBreathingParams()
    index = pd.date_range('2020-01-01', periods=21, freq='B')
    data = pd.DataFrame({'open': 100.0, 'high': 101.0, 'low': 99.0, 'close': 100.0, 'volume': 1000.0}, index=index)
    forced = pd.DataFrame({'breathing': [0.8, 0.8] + [-0.8] * 19, 'trend': [0.1, 0.1] + [-0.1] * 19}, index=index)
    ctx = StrategyContext(symbol='SPY', timeframe='1d', warmup_bars=strategy.warmup_bars(params))
    result = strategy.generate_signals(data, forced, ctx, params).data
    signal = result['signal'].tolist()
    assert signal[:2] == [0, 0]
    assert signal[2:10] == [1] * 8
    assert signal[10] == 0
    assert signal[11:19] == [-1] * 8
    assert signal[19] == 0
    assert result['size'].gt(0.0).all()
    flat = strategy.indicators(data, params)
    assert strategy.generate_signals(data, flat, ctx, params).data['signal'].eq(0).all()
