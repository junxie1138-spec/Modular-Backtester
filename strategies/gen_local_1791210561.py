from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class OrdinalScaleParams:
    pattern_window: int = 24
    entropy_gap: float = 0.08
    ordinal_threshold: float = 0.12
    efficiency_window: int = 8
    efficiency_threshold: float = 0.15
    volatility_window: int = 20
    target_volatility: float = 0.12
    holding_bars: int = 8

    def __post_init__(self):
        if min(self.pattern_window, self.efficiency_window, self.volatility_window) < 2:
            raise ValueError('Lookbacks must be at least two bars.')
        if self.holding_bars < 1:
            raise ValueError('holding_bars must be positive.')
        for value in (self.entropy_gap, self.ordinal_threshold, self.efficiency_threshold):
            if not np.isfinite(value) or not 0.0 < value < 1.0:
                raise ValueError('Signal thresholds must be finite and between zero and one.')
        if not np.isfinite(self.target_volatility) or self.target_volatility <= 0.0:
            raise ValueError('target_volatility must be finite and positive.')


class GeneratedStrategy(BaseStrategy[OrdinalScaleParams]):
    strategy_id = 'gen_local_1791210561'

    @classmethod
    def params_type(cls):
        return OrdinalScaleParams

    def warmup_bars(self, params):
        return int(max(params.pattern_window + 6, params.efficiency_window, params.volatility_window + 1))

    def indicators(self, data, params):
        out = pd.DataFrame(index=data.index)
        close = data['close'].where(np.isfinite(data['close']) & (data['close'] > 0.0))
        opening = data['open'].where(np.isfinite(data['open']) & (data['open'] > 0.0))
        out['log_return'] = np.log(close).diff()
        out['intraday_return'] = np.log(close / opening)

        for lag, label in ((1, 'fast'), (3, 'slow')):
            a, b, c = close.shift(2 * lag), close.shift(lag), close
            valid = a.notna() & b.notna() & c.notna()
            # Equal prices receive chronological ranks, making ties deterministic.
            code = 4 * (b >= a).astype(int) + 2 * (c >= a).astype(int) + (c >= b).astype(int)
            out[label + '_pattern'] = code.where(valid)
            columns = []
            for pattern in (0, 1, 3, 4, 6, 7):
                column = label + '_p' + str(pattern)
                out[column] = (code == pattern).astype(float).where(valid).rolling(params.pattern_window, min_periods=params.pattern_window).mean()
                columns.append(column)
            probabilities = out[columns]
            out[label + '_entropy'] = -(probabilities * np.log(probabilities.clip(lower=1e-12))).sum(axis=1, min_count=6) / np.log(6.0)

        out['complexity_separation'] = out['slow_entropy'] - out['fast_entropy']
        out['ordinal_balance'] = out['fast_p7'] - out['fast_p0']
        out['intraday_sum'] = out['intraday_return'].rolling(params.efficiency_window, min_periods=params.efficiency_window).sum()
        out['intraday_absolute_sum'] = out['intraday_return'].abs().rolling(params.efficiency_window, min_periods=params.efficiency_window).sum()
        out['intraday_efficiency'] = out['intraday_sum'] / out['intraday_absolute_sum'].replace(0.0, np.nan)
        out['annual_volatility'] = out['log_return'].rolling(params.volatility_window, min_periods=params.volatility_window).std(ddof=0) * np.sqrt(252.0)
        out['target_size'] = (params.target_volatility / out['annual_volatility'].clip(lower=0.01)).clip(lower=0.05, upper=0.95)

        orderly = out['complexity_separation'] >= params.entropy_gap
        ordinal_long = orderly & (out['ordinal_balance'] >= params.ordinal_threshold)
        ordinal_short = orderly & (out['ordinal_balance'] <= -params.ordinal_threshold)
        intraday_long = out['intraday_efficiency'] >= params.efficiency_threshold
        intraday_short = out['intraday_efficiency'] <= -params.efficiency_threshold
        ready = np.isfinite(out[['complexity_separation', 'ordinal_balance', 'intraday_efficiency', 'target_size']]).all(axis=1)
        out['entry_direction'] = np.select([ready & ordinal_long & intraday_long, ready & ordinal_short & intraday_short], [1, -1], default=0).astype(int)
        return out

    def generate_signals(self, data, indicators, ctx: StrategyContext, params):
        directions = indicators['entry_direction'].fillna(0).to_numpy(dtype=int)
        exposures = indicators['target_size'].to_numpy(dtype=float)
        signals = np.zeros(len(data), dtype=int)
        sizes = np.ones(len(data), dtype=float)
        position = 0
        entry_bar = -1
        entry_size = 1.0
        for i in range(len(data)):
            if position and i - entry_bar >= params.holding_bars:
                position = 0
                # Leave this decision bar flat; another entry can occur next bar.
            elif position == 0 and directions[i] != 0 and np.isfinite(exposures[i]) and exposures[i] > 0.0:
                position = int(directions[i])
                entry_bar = i
                entry_size = float(exposures[i])
            signals[i] = position
            if position:
                sizes[i] = entry_size
        df = pd.DataFrame({'signal': signals, 'size': sizes}, index=data.index)
        df['signal'] = df['signal'].shift(1).fillna(0).astype(int)
        df['size'] = df['size'].shift(1).fillna(1.0).astype(float)
        return SignalFrame(data=df, signal_column='signal', size_column='size')


if __name__ == '__main__':
    params = OrdinalScaleParams()
    strategy = GeneratedStrategy()
    index = pd.date_range('2020-01-01', periods=22, freq='B')
    data = pd.DataFrame(index=index)
    indicators = pd.DataFrame({'entry_direction': [1] + [-1] * 21, 'target_size': np.linspace(0.2, 0.8, 22)}, index=index)
    result = strategy.generate_signals(data, indicators, None, params).data
    assert result['signal'].iloc[0] == 0
    assert (result['signal'].iloc[1:9] == 1).all()
    assert result['signal'].iloc[9] == 0
    assert (result['signal'].iloc[10:18] == -1).all()
    assert result['signal'].iloc[18] == 0
    assert np.allclose(result['size'].iloc[1:9], indicators['target_size'].iloc[0])
    assert np.allclose(result['size'].iloc[10:18], indicators['target_size'].iloc[9])
    assert (result['size'] > 0.0).all()
