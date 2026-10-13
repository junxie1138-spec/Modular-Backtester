from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class ShockAutocorrelationParams:
    volume_window: int = 20
    memory_window: int = 80
    percentile_window: int = 126
    percentile: float = 0.8
    holding_bars: int = 4

    def __post_init__(self):
        if self.volume_window < 2 or self.memory_window < 8 or self.percentile_window < 2:
            raise ValueError('Lookback windows are too short.')
        if not 0.0 < self.percentile < 1.0:
            raise ValueError('percentile must be between zero and one.')
        if not 3 <= self.holding_bars <= 5:
            raise ValueError('holding_bars must be between three and five.')


class GeneratedStrategy(BaseStrategy[ShockAutocorrelationParams]):
    strategy_id = 'gen_local_1791242182'

    @classmethod
    def params_type(cls):
        return ShockAutocorrelationParams

    def warmup_bars(self, params: ShockAutocorrelationParams) -> int:
        return int(params.volume_window + params.memory_window + params.percentile_window + 2)

    def indicators(self, data: pd.DataFrame, params: ShockAutocorrelationParams) -> pd.DataFrame:
        out = pd.DataFrame(index=data.index)
        close = data['close'].where(data['close'] > 0)
        out['return'] = close.pct_change(fill_method=None).replace([np.inf, -np.inf], np.nan)
        out['absolute_return'] = out['return'].abs()
        out['volume_threshold'] = data['volume'].shift(1).rolling(params.volume_window).quantile(params.percentile)
        out['volume_shock'] = (data['volume'] > out['volume_threshold']) & (data['volume'] > 0)
        out['preceding_volume_shock'] = out['volume_shock'].shift(1, fill_value=False)
        out['shock_return'] = out['return'].shift(1).where(out['preceding_volume_shock'])
        out['response_return'] = out['return'].where(out['preceding_volume_shock'])
        out['autocorrelation'] = out['shock_return'].rolling(params.memory_window, min_periods=8).corr(out['response_return']).shift(1).replace([np.inf, -np.inf], np.nan)
        out['correlation_strength'] = out['autocorrelation'].abs()
        out['strength_threshold'] = out['correlation_strength'].shift(1).rolling(params.percentile_window).quantile(params.percentile)
        out['return_threshold'] = out['absolute_return'].shift(1).rolling(params.percentile_window).quantile(params.percentile)
        eligible = out['volume_shock'] & (out['absolute_return'] > out['return_threshold']) & (out['correlation_strength'] > out['strength_threshold'])
        out['entry'] = (np.sign(out['return']) * np.sign(out['autocorrelation'])).where(eligible, 0).fillna(0).astype(int)
        return out

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: ShockAutocorrelationParams) -> SignalFrame:
        entries = indicators['entry'].fillna(0).to_numpy(dtype=int)
        signals = np.zeros(len(data), dtype=int)
        position = 0
        exit_bar = -1
        warmup = self.warmup_bars(params)
        for i, entry in enumerate(entries):
            if position:
                if i == exit_bar:
                    position = 0
            elif i >= warmup and entry:
                position = int(entry)
                exit_bar = i + params.holding_bars
            signals[i] = position
        df = pd.DataFrame({'signal': signals, 'size': 1.0}, index=data.index)
        df['signal'] = df['signal'].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column='signal', size_column='size')


def _self_check():
    strategy = GeneratedStrategy()
    params = ShockAutocorrelationParams()
    warmup = strategy.warmup_bars(params)
    index = pd.date_range('2020-01-01', periods=warmup + 12)
    data = pd.DataFrame({'open': 100.0, 'high': 101.0, 'low': 99.0, 'close': 100.0, 'volume': 1000.0}, index=index)
    derived = strategy.indicators(data, params)
    assert derived.index.equals(data.index)
    assert derived['entry'].eq(0).all()
    entries = np.zeros(len(data), dtype=int)
    entries[warmup:] = -1
    entries[warmup] = 1
    forced = pd.DataFrame({'entry': entries}, index=index)
    ctx = StrategyContext(symbol='SPY', timeframe='1d', warmup_bars=warmup)
    result = strategy.generate_signals(data, forced, ctx, params).data
    assert result['signal'].iloc[:warmup + 1].eq(0).all()
    assert result['signal'].iloc[warmup:warmup + 11].tolist() == [0, 1, 1, 1, 1, 0, -1, -1, -1, -1, 0]
    assert result['size'].gt(0).all()


if __name__ == '__main__':
    _self_check()
